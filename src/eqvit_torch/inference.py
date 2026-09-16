from dataclasses import dataclass
import numpy as np, torch
from .preprocess import standardize_for_picker, amplitude_preserving_mag, extract_p_window

def pick_p(model,x,fs=100.,threshold=.1,device='cpu'):
    z=standardize_for_picker(x); t=torch.from_numpy(z[None]).to(device); model.eval()
    with torch.no_grad(): pr=model(t)[0].detach().cpu().numpy()
    candidates=np.flatnonzero(pr>=threshold); idx=int(candidates[np.argmax(pr[candidates])]) if len(candidates) else int(np.argmax(pr))
    return {'sample':idx,'seconds':idx/fs,'probability':float(pr[idx]),'probabilities':pr,'triggered':bool(pr[idx]>=threshold)}

def estimate_magnitude(model,x,p_sample,device='cpu',samples=3000,pre=100):
    w=amplitude_preserving_mag(extract_p_window(x,p_sample,samples,pre)); model.eval()
    with torch.no_grad(): y=model(torch.from_numpy(w[None]).to(device)).item()
    return float(y),w

def mc_magnitude(model,x,p_sample,device='cpu',runs=50,samples=3000,pre=100):
    w=amplitude_preserving_mag(extract_p_window(x,p_sample,samples,pre)); model.train(); vals=[]
    with torch.no_grad():
        q=torch.from_numpy(w[None]).to(device)
        for _ in range(runs): vals.append(float(model(q).item()))
    return {'mean':float(np.mean(vals)),'std':float(np.std(vals)),'samples':np.asarray(vals),'window':w}

@dataclass
class EarlyWarningPipeline:
    picker: object; magnitude: object; fs: float=100.; threshold: float=.1; device: str='cpu'; magnitude_samples:int=3000
    def process_window(self,x):
        p=pick_p(self.picker,x,self.fs,self.threshold,self.device)
        if not p['triggered']: return {'pick':p,'magnitude':None}
        m,w=estimate_magnitude(self.magnitude,x,p['sample'],self.device,self.magnitude_samples,100)
        return {'pick':p,'magnitude':m,'magnitude_window':w}
