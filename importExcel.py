import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
ss=0.5
w=0.7
batch=120
halfbatch=60
filename=r"./2021-51MCM-Problem C/附件1(Appendix 1)2021-51MCM-Problem C.xlsx"
f= pd.read_excel(filename)
valWithNum=np.array(f)
valWithTime=valWithNum[:,1:]
time=valWithTime[:,0]
valNoStd=valWithTime[:,1:]#value without stanrdardzation
judge=abs(valNoStd.max(0)-valNoStd.min(0))<0.0001#constant judge

val=preprocessing.scale(valNoStd)
val=torch.Tensor(val)
prob_tor=torch.Tensor(5519,100)

prob_tor[0:halfbatch,:]= torch.Tensor(preprocessing.scale(val[0:batch,:]))[0:halfbatch,:]
prob_tor[-halfbatch:,:]=torch.Tensor(preprocessing.scale(val[-batch:,:]))[-halfbatch:,:]
for i in range(halfbatch,5519-halfbatch+1):
    prob_tor[i,:]=torch.Tensor(preprocessing.scale(val[i-halfbatch:i+halfbatch,:]))[halfbatch,:]
prob_tor=abs(torch.tanh(prob_tor))

#5 points smooth
prob5s=torch.Tensor(5519,100)
prob5s[0,:]=prob_tor[0,:]-ss*(prob_tor[0,:]+prob_tor[0,:]+prob_tor[0+1,:]+prob_tor[0+2,:]-4*prob_tor[0,:])/4
prob5s[1,:]=prob_tor[1,:]-ss*(prob_tor[0,:]+prob_tor[0,:]+prob_tor[2,:]+prob_tor[3,:]-4*prob_tor[1,:])/4
prob5s[5517,:]=prob_tor[5517,:]-ss*(prob_tor[5515,:]+prob_tor[5516,:]+prob_tor[5518,:]+prob_tor[5518,:]-4*prob_tor[5517,:])/4
prob5s[5518,:]=prob_tor[5518,:]-ss*(prob_tor[5516,:]+prob_tor[5517,:]+prob_tor[5518,:]+prob_tor[5518,:]-4*prob_tor[5518,:])/4
varnum=100-judge.sum()
for i in range(2,5517):
    prob5s[i,:]=prob_tor[i,:]-ss*(prob_tor[i-2,:]+prob_tor[i-1,:]+prob_tor[i+1,:]+prob_tor[i+2,:]-4*prob_tor[i,:])/4
for j in range(100):
    if judge[j]:
        prob5s[:,j]=0
prob5s=w*prob5s+(1-w)*prob5s.sum(1)[0].reshape(-1,1)/varnum
prob5s=abs(torch.tanh(prob5s/2))
prob5s=prob5s/prob5s.max()
prob5s_np=prob5s.numpy()

a=1