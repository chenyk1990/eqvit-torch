import argparse,h5py,numpy as np,matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--h5',required=True);p.add_argument('--ids',required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
ids=np.load(a.ids,allow_pickle=True); sig=[x for x in ids.astype(str) if x.endswith('_EV')]; key=sig[a.index]
with h5py.File(a.h5,'r') as f:
 g=f[key]; x=np.asarray(g['data']); p0=int(g.attrs['p_arrival_sample']); m=float(g.attrs['magnitude']); print(key,'M=',m,'P=',p0);
 for j,n in enumerate('ZNE'): plt.plot(np.arange(len(x))/100,x[:,j]/(np.max(np.abs(x[:,j]))+1e-12)+j*2,label=n)
 plt.axvline(p0/100,ls='--');plt.xlabel('Time (s)');plt.title(f'{key}  M={m:.2f}');plt.show()
