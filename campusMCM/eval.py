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
data=[0.02941,0.03904,0.03738]
data_x=torch.Tensor(data).reshape(-1,1,3)
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        x=self.reg(x)
        return x
net = lstm_reg(3, 100)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

net.load_state_dict(torch.load("lstm8.pth"))
net.eval()
net.to(c)
pre=[]
cc=data_x[:,:,-2:]
for i in range(100):
    y=net(data_x)
    a2=data_x[:, :, 1]
    a3=data_x[:, :, 2]
    data_x[:,:,0]=a2
    data_x[:,:,1]=a3
    data_x[:,:,2]=y
    pre.append(y)
pre=torch.Tensor(pre)
pre=pre.numpy()


a=1