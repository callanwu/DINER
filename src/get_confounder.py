from model import ABSAmodel
from transformers import AutoTokenizer,AutoModel
import torch
from data_utills import *
from torch import nn
from tqdm import tqdm

class ConfounderModel(nn.Module):
    def __init__(self):
        super(ConfounderModel, self).__init__()
        self.backbone = ABSAmodel()
        self.backbone.load_state_dict(torch.load(""))
        self.pooler =  nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        )
    def forward(self, text_input_ids, text_attention_mask):
        extended_attention_mask: torch.Tensor = self.backbone.bert.get_extended_attention_mask(text_attention_mask, text_input_ids.size())
        embeddings = self.backbone.bert.embeddings(input_ids=text_input_ids)
        hidden_states = embeddings
        for i in range(3):
            layer = self.backbone.bert.encoder.layer[i]
            layer_outputs = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        cls_output = hidden_states[:, 0, :]
        pooler_output = self.pooler(cls_output)
        return pooler_output

model = ConfounderModel()
cfer = torch.zeros((983,768)).cuda()
train,test,dev = load_data("laptop","ARTS")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
train_data_loader = create_data_loader(train, tokenizer, max_len = 120, max_len_a= 13,batch_size=256)
device = torch.device("cuda:0")
model.to(device)
model.eval()
aspect_to_id = {}
num = 0
num_classes = torch.zeros((983)).cuda()
with torch.no_grad():
    for data in tqdm(train_data_loader):
        text_input_ids = data['text_input_ids'].to(device)
        text_attention_mask = data['text_attention_mask'].to(device)
        aspect_input_ids = data["aspect_input_ids"].to(device) 
        outputs = model(
                text_input_ids,text_attention_mask
            )
        for aspect,representation in zip(tokenizer.batch_decode(aspect_input_ids,skip_special_tokens=True),outputs):
            if aspect in aspect_to_id:
                cfer[aspect_to_id[aspect]] += representation
                num_classes[aspect_to_id[aspect]] += 1
            else:
                aspect_to_id[aspect] = num
                cfer[aspect_to_id[aspect]] += representation
                num_classes[aspect_to_id[aspect]] += 1
                num+=1
cfer = cfer / num_classes[:,None]
torch.save(cfer, 'memory.pt')