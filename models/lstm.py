import torch 
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward propagate RNN
        h, c = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
             Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        out, _ = self.lstm(x, (h, c))
        
        # Decode hidden state of last time step
        # out = self.fc(out[:, -1, :])  
        out = self.fc(out)  
        # out = self.fc(out[-1, :, :])  
        return out

    # def reset(self):
    #     # Set initial states 
    #     # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
    #     # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
    #     self.h = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
    #               Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
    #
