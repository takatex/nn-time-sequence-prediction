import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.features = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 2),
            nn.MaxPool1d(3),
            nn.Conv1d(hidden_size, hidden_size, 2),
            nn.MaxPool1d(2),
            nn.ReLU()
            )
        # self.c1 = nn.Conv1d(input_size, hidden_size, 2)
        # self.p1 = nn.MaxPool1d(3)
        # self.c2 = nn.Conv1d(hidden_size, hidden_size, 2)
        # self.p2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(0, 1).transpose(1, 2)
        out = self.features(x)
        # Run through Conv1d and Pool1d layers
        # c = self.c1(inputs)
        # print(c.size())
        # p = self.p1(c)
        # print(p.size())
        # c = self.c2(p)
        # print(c.size())
        # p = self.p2(c)
        # print(p.size())
        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        out = out.transpose(1, 2).transpose(0, 1)
        out = self.fc(out)
        return out
