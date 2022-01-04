import torch
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
import pandas as pd
from torch import nn
from torch.autograd import Variable
d=torch.device("cuda:0")
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(0,len(dataset)-look_back):
        a = dataset[i:(i + look_back)]
        b = dataset[(i + look_back)]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)
filename=r"D:\2021年度兰州大学数学建模竞赛赛题\温州 (1).xlsx"
f= pd.read_excel(filename)
f=f.values
data=f[0:28,9]

data_X, data_Y = create_dataset(data)
data_X=data_X.astype(np.float64)
train_x = torch.Tensor(data_X).reshape(-1,1,3)
train_y = torch.Tensor(data_Y).reshape(-1,1,1)
# 定义模型
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
# 开始训练
net.train()
net.to(d)
for e in range(1000):
    var_x = Variable(train_x).to(d)
    var_y = Variable(train_y).to(d)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 10 == 0: # 每 10 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss))

torch.save(net.state_dict(), "lstm8.pth")


a=1
