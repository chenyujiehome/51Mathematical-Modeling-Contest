import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
def dangerousAbnormality(data,len,judge,ss=0.5,w=0.7,batch=120,halfbatch=60):

    val=torch.Tensor(data)
    prob_tor=torch.Tensor(len,100)

    prob_tor[0:halfbatch,:]= torch.Tensor(preprocessing.scale(val[0:batch,:]))[0:halfbatch,:]
    prob_tor[-halfbatch:,:]=torch.Tensor(preprocessing.scale(val[-batch:,:]))[-halfbatch:,:]
    for i in range(halfbatch,len-halfbatch+1):
        prob_tor[i,:]=torch.Tensor(preprocessing.scale(val[i-halfbatch:i+halfbatch,:]))[halfbatch,:]
    prob_tor=abs(torch.tanh(prob_tor))

    #5 points smooth
    prob5s=torch.Tensor(len,100)
    prob5s[0,:]=prob_tor[0,:]-ss*(prob_tor[0,:]+prob_tor[0,:]+prob_tor[0+1,:]+prob_tor[0+2,:]-4*prob_tor[0,:])/4
    prob5s[1,:]=prob_tor[1,:]-ss*(prob_tor[0,:]+prob_tor[0,:]+prob_tor[2,:]+prob_tor[3,:]-4*prob_tor[1,:])/4
    prob5s[len-2,:]=prob_tor[len-2,:]-ss*(prob_tor[len-4,:]+prob_tor[len-3,:]+prob_tor[len-1,:]+prob_tor[len-1,:]-4*prob_tor[len-2,:])/4
    prob5s[len-1,:]=prob_tor[len-1,:]-ss*(prob_tor[len-3,:]+prob_tor[len-2,:]+prob_tor[len-1,:]+prob_tor[len-1,:]-4*prob_tor[len-1,:])/4
    varnum=100-judge.sum()
    for i in range(2,len-2):
        prob5s[i,:]=prob_tor[i,:]-ss*(prob_tor[i-2,:]+prob_tor[i-1,:]+prob_tor[i+1,:]+prob_tor[i+2,:]-4*prob_tor[i,:])/4
    for j in range(100):
        if judge[j]:
            prob5s[:,j]=0
    prob5s=w*prob5s+(1-w)*prob5s.sum(1)[0].reshape(-1,1)/varnum
    prob5s=abs(torch.tanh(prob5s/2))
    prob5s=prob5s/prob5s.max()
    for j in range(100):
        if judge[j]:
            prob5s[:,j]=0
    prob5s_np=prob5s.numpy()
    # ff=open("prob.txt","w")
    # ff.write()
    # ff.close()
    return prob5s_np