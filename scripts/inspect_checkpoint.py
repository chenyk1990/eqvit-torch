import argparse, torch
p=argparse.ArgumentParser(); p.add_argument('checkpoint'); a=p.parse_args(); o=torch.load(a.checkpoint,map_location='cpu',weights_only=False); print(type(o));
if isinstance(o,dict): print('keys:',list(o)[:30]); sd=o.get('state_dict',o.get('model_state_dict')); print('state keys:', list(sd)[:30] if isinstance(sd,dict) else 'n/a')
