import torch, numpy as np
from eqvit_torch.models import EQViTPicker,EQViTMagnitude,KerasMHA
from eqvit_torch.eew import extract_p_centered

def test_picker_shape():
 m=EQViTPicker(samples=1000,patch=40); y=m(torch.randn(2,1000,3)); assert y.shape==(2,1000) and torch.all((y>=0)&(y<=1))
def test_mag_shape():
 m=EQViTMagnitude(); assert m(torch.randn(2,3000,3)).shape==(2,)
def test_keras_mha_inner_width():
 m=KerasMHA(40,4,40); assert m.q.out_features==160 and m.o.in_features==160
def test_extract():
 x=np.ones((500,3),np.float32); y=extract_p_centered(x,50,300,100); assert y.shape==(300,3) and np.all(y[:50]==0)
