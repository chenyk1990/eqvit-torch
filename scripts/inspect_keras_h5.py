#!/usr/bin/env python
import argparse,h5py
p=argparse.ArgumentParser(); p.add_argument('file'); a=p.parse_args()
with h5py.File(a.file,'r') as f:
 def visit(n,o):
  if isinstance(o,h5py.Dataset): print(f'{n:90s} {str(o.shape):18s} {o.dtype}')
 f.visititems(visit)
