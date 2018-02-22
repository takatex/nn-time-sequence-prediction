import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchqrnn import qrnn

class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_cuda):
        super(QRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.qrnn = qrnn.QRNN(input_size, hidden_size, num_layers, use_cuda)
        self.qrnn.use_cuda = use_cuda
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, flag=False):
        # print(x.size(1))
        h = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.qrnn(x, h)
        
        # Decode hidden state of last time step
        # print(out.shape)
        # print(out[-1, :, :].shape)
        # out = self.fc(out[-1, :, :])  
        out = self.fc(out)  
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
