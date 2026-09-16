"""PyTorch models for EQViT.

`EQViTPicker` follows the released Keras picker, including the two convF1
blocks surrounding each attention layer and Keras MHA key_dim semantics.
`EQViTMagnitude` follows VIT_Magnitude_Estimation.ipynb.
"""
import torch
from torch import nn

class ConvF1(nn.Module):
    def __init__(self, cin, cout, k=11, p=.1):
        super().__init__(); pad=k//2
        self.c1=nn.Conv1d(cin,cin,k,padding=pad); self.b1=nn.BatchNorm1d(cin)
        self.c2=nn.Conv1d(cin,cin,k,padding=pad); self.b2=nn.BatchNorm1d(cin)
        self.c3=nn.Conv1d(cin,cout,k,padding=pad); self.b3=nn.BatchNorm1d(cout)
        self.a=nn.GELU(); self.d=nn.Dropout(p)
    def forward(self,x):
        y=self.a(self.b1(self.c1(x))); y=self.a(self.b2(self.c2(y)))+x
        return self.d(self.a(self.b3(self.c3(y))))

class KerasMHA(nn.Module):
    """MHA matching Keras MultiHeadAttention(num_heads=H,key_dim=D).
    Unlike torch.nn.MultiheadAttention, Keras uses H*D internal Q/K/V width.
    """
    def __init__(self, d_model, heads=4, key_dim=40, p=.1):
        super().__init__(); self.h=heads; self.d=key_dim; inner=heads*key_dim
        self.q=nn.Linear(d_model,inner); self.k=nn.Linear(d_model,inner); self.v=nn.Linear(d_model,inner)
        self.o=nn.Linear(inner,d_model); self.drop=nn.Dropout(p)
    def forward(self,x):
        b,n,_=x.shape; h,d=self.h,self.d
        q=self.q(x).view(b,n,h,d).transpose(1,2); k=self.k(x).view(b,n,h,d).transpose(1,2); v=self.v(x).view(b,n,h,d).transpose(1,2)
        a=torch.softmax((q@k.transpose(-2,-1))/(d**.5),dim=-1); a=self.drop(a)
        return self.o((a@v).transpose(1,2).contiguous().view(b,n,h*d))

class PickerTransformerBlock(nn.Module):
    def __init__(self,dim=40,heads=4,key_dim=40,p=.1):
        super().__init__(); self.pre=ConvF1(dim,dim,11,p); self.n1=nn.LayerNorm(dim,eps=1e-6)
        self.att=KerasMHA(dim,heads,key_dim,p); self.post=ConvF1(dim,dim,11,p)
        self.n2=nn.LayerNorm(dim,eps=1e-6); self.ff=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Dropout(p),nn.Linear(2*dim,dim),nn.Dropout(p))
    def forward(self,x):
        # conv1d operates on channels-first
        z=self.pre(x.transpose(1,2)).transpose(1,2); y=self.post(self.att(self.n1(z)).transpose(1,2)).transpose(1,2)
        x=z+y; return x+self.ff(self.n2(x))

class Encoder(nn.Module):
    def __init__(self,d=100,heads=4,key_dim=100,p=.1):
        super().__init__(); self.n1=nn.LayerNorm(d,eps=1e-6); self.att=KerasMHA(d,heads,key_dim,p); self.n2=nn.LayerNorm(d,eps=1e-6)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(p),nn.Linear(2*d,d),nn.Dropout(p))
    def forward(self,x):
        x=x+self.att(self.n1(x)); return x+self.ff(self.n2(x))

class EQViTPicker(nn.Module):
    def __init__(self,samples=6000,patch=40,dim=40,heads=4,layers=4):
        super().__init__(); assert samples%patch==0
        self.samples=samples; self.patch=patch
        self.front=nn.Sequential(ConvF1(3,10),ConvF1(10,20),ConvF1(20,40))
        self.patch_proj=nn.Linear(patch*40,dim); self.pos=nn.Parameter(torch.zeros(1,samples//patch,dim))
        self.blocks=nn.ModuleList([PickerTransformerBlock(dim,heads,dim,.1) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim,eps=1e-6); self.drop=nn.Dropout(.1); self.out=nn.Conv1d(1,1,15,padding=7)
    def forward(self,x):
        if x.ndim!=3: raise ValueError('Expected B,T,3 or B,3,T')
        if x.shape[-1]==3:x=x.transpose(1,2)
        x=self.front(x); b,c,t=x.shape
        x=x.transpose(1,2).reshape(b,t//self.patch,self.patch*c); x=self.patch_proj(x)+self.pos[:,:t//self.patch]
        for blk in self.blocks:x=blk(x)
        x=self.drop(self.norm(x)).reshape(b,1,-1)
        return torch.sigmoid(self.out(x)).squeeze(1)

class MagConv(nn.Module):
    def __init__(self,cin,cout,pool,p=.2):
        super().__init__(); self.c=nn.Conv1d(cin,cout,3,padding=1); self.d=nn.Dropout(p); self.pool=nn.MaxPool1d(pool,ceil_mode=True)
    def forward(self,x): return self.pool(self.d(self.c(x)))

class EQViTMagnitude(nn.Module):
    def __init__(self,samples=3000,patch=5,dim=100,heads=4,layers=4,drop=.2,adaptive=False):
        super().__init__(); self.samples=samples; self.patch=patch; self.adaptive=adaptive
        self.conv=nn.Sequential(MagConv(3,64,2,drop),MagConv(64,32,2,drop),MagConv(32,32,2,drop),MagConv(32,32,5,drop))
        self.to75=nn.AdaptiveMaxPool1d(75) if adaptive else nn.Identity(); tokens=15 if (samples==3000 or adaptive) else max(1,((samples+39)//40)//patch)
        self.proj=nn.Linear(patch*32,dim); self.pos=nn.Parameter(torch.zeros(1,tokens,dim)); self.blocks=nn.ModuleList([Encoder(dim,heads,dim,.1) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim,eps=1e-6); self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.5),nn.Linear(tokens*dim,1000),nn.GELU(),nn.Dropout(.5),nn.Linear(1000,500),nn.GELU(),nn.Dropout(.5),nn.Linear(500,1))
    def forward(self,x):
        if x.shape[-1]==3:x=x.transpose(1,2)
        x=self.to75(self.conv(x)); b,c,t=x.shape
        if t%self.patch: raise ValueError(f'post-conv length {t} not divisible by patch {self.patch}')
        x=x.transpose(1,2).reshape(b,t//self.patch,self.patch*c); x=self.proj(x)+self.pos[:,:t//self.patch]
        for blk in self.blocks:x=blk(x)
        return self.head(self.norm(x)).squeeze(-1)

def build_picker(**kw): return EQViTPicker(**kw)
def build_magnitude(**kw): return EQViTMagnitude(**kw)
