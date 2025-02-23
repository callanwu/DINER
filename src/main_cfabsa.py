from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch import nn
from data_utills import *
from model import ABSAmodel,CFABSAmodel
from utils import *
import argparse


parser = argparse.ArgumentParser(description='CFABSA finetuning')
parser.add_argument("--ARTS",type=int)
parser.add_argument("--Counterfactual",type=int)
parser.add_argument("--GPU",type=str)
parser.add_argument("--fusion_mode",type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_len_s', type=int)
parser.add_argument('--max_len_a', type=int)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

if args.Counterfactual:
    if args.ARTS:
        train,test,dev = load_data(args.dataset_name,"ARTS")
        MODEL = CFABSAmodel(1,1)
    else:
        train,test,dev = load_data(args.dataset_name,"ORI")
        MODEL = CFABSAmodel()
else:
    if args.ARTS:
        train,test,dev = load_data(args.dataset_name,"ARTS")
        MODEL = ABSAmodel()
    else:
        train,test,dev = load_data(args.dataset_name,"ORI")
        MODEL = ABSAmodel()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda:"+args.GPU if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
my_model = MODEL.to(device)
train_data_loader = create_data_loader(train, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a,batch_size=args.batch_size)
val_data_loader = create_data_loader(dev, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a, batch_size = args.batch_size)
test_data_loader = create_data_loader(test, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a,  batch_size = args.batch_size)
total_steps = len(train_data_loader) * args.epoch

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in my_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in my_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=int(total_steps*0.2),
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
parser = argparse.ArgumentParser(description='CFABSA finetuning')
save_dir = args.save_dir
main(args.epoch,
     my_model,
     train_data_loader,
     val_data_loader,
     test_data_loader,
     loss_fn,
     optimizer,
     device,
     scheduler,
     save_dir,
     args.Counterfactual)