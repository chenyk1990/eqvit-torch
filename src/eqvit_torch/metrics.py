import numpy as np

def magnitude_metrics(y,p):
 y=np.asarray(y,float); p=np.asarray(p,float); e=p-y
 return {'n':len(y),'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(np.mean(e)),'sigma':float(np.std(e))}

def picking_metrics(true_samples,pred_samples,sr=100.):
 e=(np.asarray(pred_samples)-np.asarray(true_samples))/sr
 return {'n':len(e),'mae_s':float(np.mean(abs(e))),'bias_s':float(np.mean(e)),'sigma_s':float(np.std(e)),'within_0.2s':float(np.mean(abs(e)<=.2)),'within_0.5s':float(np.mean(abs(e)<=.5))}
