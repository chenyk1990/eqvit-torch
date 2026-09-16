from dataclasses import dataclass
import numpy as np, torch
@dataclass
class EEWEstimate:
 p_sample:int; p_probability:float; magnitude:float; magnitude_std:float|None=None; latency_s:float|None=None

def mc_magnitude(model,waveform,n=20,device='cpu'):
 x=torch.as_tensor(waveform,dtype=torch.float32,device=device)[None]
 model.train() # MC dropout, as in EQViT paper
 vals=[]
 with torch.no_grad():
  for _ in range(n): vals.append(float(model(x).item()))
 return float(np.mean(vals)),float(np.std(vals))

def extract_p_centered(x,p,n=3000,pre=100):
 x=np.asarray(x); a=p-pre; b=a+n; out=np.zeros((n,3),x.dtype); s=max(0,a); e=min(len(x),b); out[s-a:e-a]=x[s:e]; return out
