import numpy as np
import torch
a=np.loadtxt(open("prediction.csv","r"))
a_tor=torch.Tensor(a)
sum=a_tor.sum(1)

sum=torch.Tensor([1])/sum
temp=sum.max()
factor=torch.Tensor([100])/temp
score=factor*sum
score=score[5519:].unsqueeze(dim=1).numpy()
mm=[]
for i in range(48):
    if i !=47:
        s=score[120*(i+1)-1]
        mm.append(s)
    else:
        s=score[-1]
        mm.append(s)
a=1