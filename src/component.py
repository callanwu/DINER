import torch
from torch import nn
import math
class Interventional_Classifier(nn.Module):
    def __init__(self, num_classes=80, feat_dim=2048, num_head=4, tau=32.0, beta=0.03125, *args):
        super(Interventional_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 32.0 / num_head
        self.norm_scale = beta       # 1.0 / 32.0      
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.reset_parameters(self.weight)
        self.feat_dim = feat_dim
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_list = torch.split(x, self.head_dim, dim=1)
        w_list = torch.split(self.weight, self.head_dim, dim=1)
        y_list = []
        for x_, w_ in zip(x_list, w_list):
            normed_x = x_ / torch.norm(x_, 2, 1, keepdim=True)
            normed_w = w_ / (torch.norm(w_, 2, 1, keepdim=True) + self.norm_scale)
            y_ = torch.mm(normed_x * self.scale, normed_w.t())   
            y_list.append(y_)
        y = sum(y_list)
        return y

class tde_classifier(nn.Module):
    def __init__(self,num_classes,feat_dim,use_intervention=True,stagetwo=True,feat_fuse='selector'):
        super(tde_classifier,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.memory = torch.load('memory.pt')
        self.feat_fuse = feat_fuse
        self.selector = nn.Linear(self.feat_dim,self.feat_dim)
        self.stagetwo = stagetwo
        if use_intervention:
            self.context_clf = Interventional_Classifier(num_classes=983, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
            self.logit_clf = Interventional_Classifier(num_classes=num_classes, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
        else:
            self.context_clf = nn.Linear(feat_dim,num_classes)
            self.logit_clf = nn.Linear(feat_dim,num_classes)
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self,feats):
        if len(list(feats.size())) == 2:
            global_feat =  feats
            memory = self.memory
        else:
            global_feat = feats.flatten(2).mean(dim=-1)
            feats = feats.flatten(2).max(dim=-1)[0]  
            memory = self.memory
        if self.stagetwo:
            pre_logits = self.softmax(self.context_clf(global_feat))
            memory_feature = torch.mm(pre_logits,memory)
            selector = self.selector(feats.clone())
            selector = selector.tanh()
            combine_feature = feats - selector * memory_feature
            logits = self.logit_clf(combine_feature)
        else:
            logits = self.context_clf(global_feat)
        return logits