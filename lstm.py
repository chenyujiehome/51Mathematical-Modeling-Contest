import torch
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
import pandas as pd
from torch import nn
from torch.autograd import Variable
d=torch.device("cuda:0")
def create_dataset(dataset, look_back=120):
    dataX, dataY = [], []
    for i in range(0,len(dataset) - look_back-60,30):
        a = dataset[i:(i + look_back),:]
        b = dataset[(i + look_back):(i+look_back+60),:]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)
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

data_X, data_Y = create_dataset(val)
train_x = torch.Tensor(data_X).permute(0,2,1)
train_y = torch.Tensor(data_Y).permute(0,2,1)
# 定义模型
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

torch.save(net.state_dict(), "lstm3.pth")


a=1
