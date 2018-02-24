import torch 
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.batch_size = batch_size
        # self.rnn = nn.RNN(input_size, hidden_size, self.num_layers)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, flag):
        # Forward propagate RNN
        h = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        out, _ = self.rnn(x, h)
        
        # print(out.shape)
        # print(out[-1, :, :].shape)
        # out = self.fc(out[-1, :, :])  
        out = self.fc(out)  
        # print(out.shape)
        if flag:
            print(out.shape)
        return out
    
    # def reset(self):
    #     self.h = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
