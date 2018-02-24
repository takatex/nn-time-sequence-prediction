import torch 
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_cuda):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        if self.use_cuda:
            h = h.cuda()
        out, _ = self.rnn(x, h)
        out = self.fc(out[-1, :, :])
        return out

    # def reset(self):
    #     self.h = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
