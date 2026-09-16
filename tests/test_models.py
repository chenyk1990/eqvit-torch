import torch
from eqvit_torch.models import EQViTPicker,EQViTMagnitude
def test_shapes():
 assert EQViTPicker()(torch.randn(2,6000,3)).shape==(2,6000)
 assert EQViTMagnitude()(torch.randn(2,3000,3)).shape==(2,)
