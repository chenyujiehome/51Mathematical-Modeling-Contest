import numpy as np
import torch
a=np.loadtxt(open("prediction.csv","r"))
a_tor=torch.Tensor(a)
sum=a_tor.sum(1)
sum=sum.unsqueeze(dim=1).numpy()
temp=sum.max()
factor=torch.Tensor([100])/temp
score=factor*sum
sum=score.numpy()
mm=[]
data_s=[]
sen=[]
senIndex=[]
for i in range(4):
    max,maxindex=score[60*i:60*i+60].max(0)
    data_s.append(max)
    mm.append(maxindex)
    temp=a_tor[60*i:60*i+60,:]
    senmax,In=temp[maxindex,:].topk(5)
    sen.append(senmax)
    senIndex.append(In)
cc=1