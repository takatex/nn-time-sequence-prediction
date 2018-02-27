import os, sys
sys.path.append(os.pardir)
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.torchqrnn import qrnn

class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_cuda):
        super(QRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.qrnn = qrnn.QRNN(input_size, hidden_size, num_layers, window=2, bias=False, use_cuda=use_cuda)
        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
                )

    def forward(self, x):
        self.qrnn.reset()
        h = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        if self.use_cuda:
            h = h.cuda()

        out, _ = self.qrnn(x, h)
        out = self.fc(out[-1, :, :])
        return out
