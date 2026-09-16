import torch

def train_epoch(model,loader,opt,device='cpu'):
    model.train(); lossfn=torch.nn.MSELoss(); total=0.; n=0
    for x,y,_ in loader:
        x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); pred=model(x); loss=lossfn(pred,y); loss.backward(); opt.step(); total+=loss.item()*len(x); n+=len(x)
    return total/max(n,1)
@torch.no_grad()
def evaluate(model,loader,device='cpu'):
    model.eval(); ys=[]; ps=[]
    for x,y,_ in loader: ys.append(y); ps.append(model(x.to(device)).cpu())
    y=torch.cat(ys); p=torch.cat(ps); e=p-y
    return {'mae':e.abs().mean().item(),'bias':e.mean().item(),'std':e.std().item(),'y':y.numpy(),'pred':p.numpy()}
