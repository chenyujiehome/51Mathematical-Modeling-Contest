import torch
import torch.nn as nn
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