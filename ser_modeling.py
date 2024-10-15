from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torchscale.component.multiway_network import MutliwayEmbedding
from torchscale.component.embedding import PositionalEmbedding
from torchscale.architecture.encoder import Encoder
from torchscale.architecture.config import EncoderConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from typing import Optional
from transformers import BertModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig

@dataclass
class SEROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BEiT3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_config = EncoderConfig(**config.encoder)
        assert self.encoder_config.multiway
        assert not self.encoder_config.share_encoder_input_output_embed
        
        self.text_embed = BertModel.from_pretrained(config.text.ckpt)
        if config.text.freeze:
            for param in self.text_embed.parameters():
                param.require_grads = False
        self.text_project_in = nn.Linear(config.text.input_dim, self.encoder_config.encoder_embed_dim)
        
        audio_config = WhisperConfig.from_pretrained(config.audio.ckpt)
        self.audio_embed = WhisperEncoder(audio_config)
        if config.audio.freeze:
            self.audio_embed._freeze_parameters()
        self.audio_project_in = nn.Linear(config.audio.input_dim, self.encoder_config.encoder_embed_dim)
        
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(config.audio.seq_len + 2, self.encoder_config.encoder_embed_dim),
                PositionalEmbedding(self.encoder_config.max_source_positions, self.encoder_config.encoder_embed_dim),
            ],
            dim=1,
        )
        
        self.encoder = Encoder(
            self.encoder_config,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(self, textual_tokens, audio_tokens, padding_mask):
        x1 = self.audio_embed(audio_tokens).last_hidden_state
        x1 = self.audio_project_in(x1)
        multiway_split_position = x1.size(1)
        x2 = self.text_embed(textual_tokens).last_hidden_state
        x2 = self.text_project_in(x2)
        x = torch.cat([x1, x2], dim=1)

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=padding_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position
        )
        encoder_out["multiway_split_position"] = multiway_split_position
        return encoder_out
    
class BEiT3Wrapper(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.beit3 = BEiT3(config)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

class BEiT3ForSER(BEiT3Wrapper):
    def __init__(self, config):
        super(BEiT3ForSER, self).__init__(config=config)
        embed_dim = config.encoder.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=nn.LayerNorm,
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2), 
            nn.GELU(),
            nn.Linear(embed_dim * 2, config.classifier.num_classes), 
        )
        self.head.apply(self._init_weights)

    def forward(self, audio, text, padding_mask, labels=None, **kwargs):
        outputs = self.beit3(
            textual_tokens=text, 
            audio_tokens=audio, 
            padding_mask=padding_mask, 
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        logits = self.head(cls_rep)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return SEROutput(
            loss=loss,
            logits=logits,
        )