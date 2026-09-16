"""Small compatibility layer: EQCCT-torch APIs can evolve without coupling the trainer."""
import numpy as np

def picks_to_cache(ids,picks,probs=None,path='eqcct_picks.npz'):
 ids=np.asarray(ids); picks=np.asarray(picks,int); probs=np.ones(len(ids)) if probs is None else np.asarray(probs,float)
 np.savez(path,ids=ids,p_samples=picks,p_prob=probs); return path

def load_pick_cache(path):
 z=np.load(path,allow_pickle=True); return {str(i):(int(p),float(q)) for i,p,q in zip(z['ids'],z['p_samples'],z['p_prob'])}
