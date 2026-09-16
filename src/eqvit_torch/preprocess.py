import numpy as np

def standardize_for_picker(x,eps=1e-6):
    x=np.asarray(x,np.float32); return (x-x.mean(0,keepdims=True))/(x.std(0,keepdims=True)+eps)

def amplitude_preserving_mag(x):
    """Magnitude input intentionally remains in physical/raw amplitude scale; only de-mean."""
    x=np.asarray(x,np.float32); return x-x.mean(0,keepdims=True)

def extract_p_window(x,p_sample,n=3000,pre=100):
    start=int(p_sample)-pre; y=np.zeros((n,3),np.float32); a=max(0,start); b=min(len(x),start+n); y[a-start:b-start]=x[a:b]; return y
