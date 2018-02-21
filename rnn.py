import torch 
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        # self.rnn = nn.RNN(input_size, hidden_size, self.num_layers)
        self.rnn = nn.RNN(1, hidden_size, self.num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, flag=False):
        # print(x.size(1))
        self.h = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.rnn(x, self.h)
        
        # Decode hidden state of last time step
        # print(out.shape)
        # print(out[-1, :, :].shape)
        out = self.fc(out[-1, :, :])  
        # out = self.fc(out)  
        # print(out.shape)
        if flag:
            print(out.shape)
        return out
    
    # def reset(self):
    #     # Set initial states 
    #     # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
    #     # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
    #     # self.h = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
    #     self.h = Variable(torch.zeros(self.num_layers, 100, self.hidden_size))
