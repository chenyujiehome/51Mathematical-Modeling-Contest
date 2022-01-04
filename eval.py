import torch
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
import pandas as pd
from torch import nn
from torch.autograd import Variable
from abnormal import dangerousAbnormality
d=torch.device("cuda:0")
c=torch.device("cpu")#CPU
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=60, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        x=self.reg(x)
        return x
net = lstm_reg(120, 200)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# 开始训练
net.train()
net.to(c)

net.load_state_dict(torch.load("lstm3.pth"))
filename=r"./2021-51MCM-Problem C/附件1(Appendix 1)2021-51MCM-Problem C.xlsx"
f= pd.read_excel(filename)
valWithNum=np.array(f)
valWithTime=valWithNum[:,1:]
time=valWithTime[:,0]
valNoNor=valWithTime[:,1:]#value without valNoNor
judge=abs(valNoNor.max(0)-valNoNor.min(0))<0.0001#constant judge
val=np.zeros((5519,100))
for j in range(100):
    if judge[j]==False:
        max_value = np.max(valNoNor[:,j])
        min_value = np.min(valNoNor[:,j])
        scalar = max_value - min_value
        val[:,j] = list(map(lambda x: (x - min_value) / scalar, valNoNor[:,j]))
data_x=val[-120:,:]
data_x=torch.Tensor(data_x).permute(1,0)
data_x=data_x.unsqueeze(dim=0)
net.eval()
net.to(c)
pre=torch.Tensor([])
for i in range(4):
    y=net(data_x)

    data_x[:,:,:60]=data_x[:,:,-60:]
    data_x[:,:,-60:]=y
    pre=torch.cat((pre,y.squeeze()),dim=1)
pre=pre.permute(1,0)
pre=pre.detach().numpy()
pre=torch.Tensor(pre)
for j in range(100):
    if judge[j]:
        pre[:,j]=0
val_tor=torch.Tensor(val)
data=torch.cat((val_tor,pre),0)
data_np=data.numpy()
prob_np=dangerousAbnormality(data_np,5759,judge)
mm=[]
for ss in range(60,80,1):
    ss=ss/100
    change=dangerousAbnormality(data_np,5759,judge,w=ss)
    loss=abs(change-ss).sum()
    mm.append(loss)


a=1