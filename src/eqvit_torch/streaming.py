import numpy as np
class RingBuffer3C:
    def __init__(self,samples): self.samples=samples; self.data=np.empty((0,3),np.float32)
    def append(self,x): self.data=np.concatenate([self.data,np.asarray(x,np.float32)],0)[-self.samples:]
    @property
    def ready(self): return len(self.data)==self.samples

def obspy_stream_to_array(stream,fs=100.):
    st=stream.copy(); st.merge(method=1,fill_value='interpolate'); st.resample(fs)
    # Prefer Z,N,E; fall back to available three channels.
    traces=[]
    for suffix in ['Z','N','E']:
        s=st.select(channel=f'*{suffix}'); traces.append(s[0].data if s else None)
    if any(v is None for v in traces): traces=[tr.data for tr in st[:3]]
    n=min(map(len,traces)); return np.stack([np.asarray(v[:n],np.float32) for v in traces],1)
