import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torchvision
import copy
       
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class PrtAttLayer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)

    def prt_interact(self,sem_prt):
        tgt2 = self.self_attn(sem_prt, sem_prt, value=sem_prt)[0]
        sem_prt = sem_prt + self.dropout1(tgt2)
        return sem_prt
        
    def prt_assign(self, vis_prt, vis_query):
        vis_prt = self.multihead_attn(query=vis_prt,
                                   key=vis_query,
                                   value=vis_query)[0]
        return vis_prt
        
    def prt_refine(self, vis_prt):
        new_vis_prt = self.linear2(self.activation(self.linear1(vis_prt)))
        return new_vis_prt + vis_prt

    def forward(self, vis_prt, vis_query):
        # sem_prt: 196*bs*c
        # vis_query: wh*bs*c
        vis_prt = self.prt_assign(vis_prt,vis_query)
        vis_prt = self.prt_refine(vis_prt)
        return vis_prt

class PrtClsLayer(nn.Module):
    def __init__(self, nc, na, dim):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        self.fc1 = nn.Linear(dim, dim//na) 
        self.fc2 = nn.Linear(dim//na, dim)
        
        self.weight_bias = nn.Parameter(torch.empty(nc, dim))
        nn.init.kaiming_uniform_(self.weight_bias, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.empty(nc))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_bias)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.activation = nn.ReLU()

    def prt_refine(self, prt):
        w = F.sigmoid(self.fc2(self.activation(self.fc1(prt))))
        prt = self.linear2(self.activation(self.linear1(prt)))
        prt = self.weight_bias + prt * w
        return prt

    def forward(self,query,cls_prt):
        cls_prt = self.prt_refine(cls_prt)
        logit = F.linear(query, cls_prt, self.bias)
        return logit,cls_prt
        
class DPN(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(DPN, self).__init__()
        ''' default '''
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size       
        vis_prt_dim = sf_size           
        self.sf =  torch.from_numpy(args.sf)
        self.args  = args
        vis_emb_dim = args.vis_emb_dim
        att_emb_dim = args.att_emb_dim
        self.args.hidden_dim=2048
        ''' backbone net'''
        if args.backbone=='resnet101':
            self.backbone = torchvision.models.resnet101()
        elif args.backbone=='resnet50':
            self.backbone = torchvision.models.resnet50()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
        ''' ZSR '''                                      
        self.vis_proj = nn.Sequential(
            nn.Conv2d(2048, vis_emb_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(vis_emb_dim),
            nn.ReLU(inplace=True),
        )
        emb_dim = att_emb_dim*vis_prt_dim                            
        # att prt
        self.zsl_prt_emb = nn.Embedding(vis_prt_dim, vis_emb_dim)
        self.zsl_prt_dec = _get_clones(PrtAttLayer(dim=vis_emb_dim, nhead=8), args.n_dec)
        self.zsl_prt_s2v = _get_clones(nn.Sequential(nn.Linear(vis_emb_dim,att_emb_dim),nn.LeakyReLU()), args.n_dec)
        # cate prt
        self.cls_prt_emb = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.kaiming_uniform_(self.cls_prt_emb, a=math.sqrt(5))
        self.cls_prt_dec = _get_clones(PrtClsLayer(nc=num_classes, na=att_emb_dim, dim=emb_dim), args.n_dec)
        # sem proj
        self.sem_proj = nn.Sequential(
            nn.Linear(sf_size,int(emb_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(emb_dim/2),emb_dim),
            nn.LeakyReLU(),
        )  
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrained:
            if args.backbone=='resnet101':
                self.backbone.load_state_dict(torch.load(args.resnet_pretrain))
            elif args.backbone=='resnet50':
                self.backbone.load_state_dict(torch.load(args.resnet_pretrain))            
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        # backbone
        last_conv = self.backbone(x)
        bs, c, h, w = last_conv.shape      
        # wh*bs*c
        vis_query = self.vis_proj(last_conv).flatten(2).permute(2, 0, 1) # wh*bs*c      
        # semantic projection
        sem_emb = self.sem_proj(self.sf.cuda())
        sem_emb_norm = F.normalize(sem_emb, p=2, dim=1)
        #attribute prototype
        vis_prt = self.zsl_prt_emb.weight.unsqueeze(1).repeat(1, bs, 1).cuda() 
        vis_embs = []
        logit_zsl = []
        for dec,proj in zip(self.zsl_prt_dec,self.zsl_prt_s2v):
            vis_prt = dec(vis_prt, vis_query)
            vis_emb = proj(vis_prt).permute(1,0,2).flatten(1) 
            vis_embs.append(vis_emb)
            vis_emb_norm = F.normalize(vis_emb, p=2, dim=1)
            logit_zsl.append(vis_emb_norm.mm(sem_emb_norm.permute(1,0))) 
        vis_embs.reverse() 
        #category prototype
        cls_prt = self.cls_prt_emb.cuda()
        logit_cls = []
        for dec,query in zip(self.cls_prt_dec,vis_embs):  
            logit,cls_prt = dec(query,cls_prt) 
            logit_cls.append(logit)       
        logit_zsl.reverse()   
        return logit_zsl,logit_cls, None 