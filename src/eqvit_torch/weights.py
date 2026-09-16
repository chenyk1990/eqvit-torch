from pathlib import Path
import torch

def _strip(sd): return {k.removeprefix('module.').removeprefix('model.'):v for k,v in sd.items()}
def load_checkpoint(model,path,device='cpu',strict=True):
    obj=torch.load(path,map_location=device,weights_only=False)
    if isinstance(obj,torch.nn.Module): model.load_state_dict(obj.state_dict(),strict=strict); return model
    sd=obj.get('state_dict',obj.get('model_state_dict',obj)) if isinstance(obj,dict) else obj
    model.load_state_dict(_strip(sd),strict=strict); return model

def load_legacy_torch_model(path,device='cpu'):
    """Load an existing Figshare Torch artifact when it is a serialized nn.Module or checkpoint.
    For unknown third-party state_dict key layouts, instantiate the matching EQViT model then call load_checkpoint(..., strict=False) and inspect missing/unexpected keys.
    """
    return torch.load(path,map_location=device,weights_only=False)
