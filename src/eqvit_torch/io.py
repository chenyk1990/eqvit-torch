from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset

def _decode(v): return v.decode() if isinstance(v,(bytes,np.bytes_)) else v
def load_txed_record(h5,key):
 g=h5[key]; return np.asarray(g['data'],dtype=np.float32),{k:_decode(v) for k,v in g.attrs.items()}
def event_id(waveform_id): return str(waveform_id).split('_')[0]

def split_ids_by_event(id_path,train=.8,val=.1,seed=2026):
 ids=np.load(id_path,allow_pickle=True).astype(str); sig=[i for i in ids if i.endswith('_EV')]; events=np.array(sorted({event_id(i) for i in sig}))
 rng=np.random.default_rng(seed); rng.shuffle(events); n=len(events); ntr=int(train*n); nv=int(val*n); tr=set(events[:ntr]); va=set(events[ntr:ntr+nv])
 return [i for i in sig if event_id(i) in tr],[i for i in sig if event_id(i) in va],[i for i in sig if event_id(i) not in tr|va]

class TXEDDataset(Dataset):
 def __init__(self,h5_path,id_path,ids=None,window_samples=3000,pre_p=100,pick_cache=None,transform=None,return_attrs=False):
  self.h5_path=str(h5_path); allids=np.load(id_path,allow_pickle=True).astype(str); self.ids=list(ids) if ids is not None else [i for i in allids if i.endswith('_EV')]
  self.window_samples=window_samples; self.pre_p=pre_p; self.transform=transform; self.return_attrs=return_attrs; self._h5=None
  self.pick_cache=pick_cache or {}
 def _file(self):
  if self._h5 is None:
   import h5py; self._h5=h5py.File(self.h5_path,'r')
  return self._h5
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  wid=self.ids[i]; x,a=load_txed_record(self._file(),wid); manual=int(float(a['p_arrival_sample'])); p,prob=self.pick_cache.get(wid,(manual,1.0))
  start=int(p)-self.pre_p; end=start+self.window_samples; out=np.zeros((self.window_samples,3),np.float32); s=max(0,start); e=min(len(x),end); out[s-start:e-start]=x[s:e]
  if self.transform: out=self.transform(out)
  meta={'id':wid,'event_id':event_id(wid),'p_arrival_sample':manual,'used_p_sample':int(p),'p_probability':float(prob),'snr_db':np.asarray(a.get('snr_db',[np.nan]*3),dtype=float)}
  if self.return_attrs: meta['attrs']=a
  return torch.from_numpy(np.ascontiguousarray(out)),torch.tensor(float(a['magnitude']),dtype=torch.float32),meta
