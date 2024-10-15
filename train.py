from ser_modeling import BEiT3ForSER
from omegaconf import OmegaConf
from transformers import BertTokenizer
import torch

config = OmegaConf.load("/Users/nngocson/Documents/FPT/SER/configs/model.yaml")
model = BEiT3ForSER(config)

audio = torch.rand(1, 80, 3000)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Replace me by any text you'd like."
text = tokenizer(text, return_tensors='pt').input_ids

model.eval()
with torch.no_grad():
    x = model(audio=audio,
         text=text,
         padding_mask=torch.zeros(1, 1500+12))
print(x)