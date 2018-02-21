import torch 
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward propagate RNN
        out, _ = self.lstm(x, self.h)  
        
        # Decode hidden state of last time step
        # out = self.fc(out[:, -1, :])  
        out = self.fc(out)  
        return out

    def reset(self):
        # Set initial states 
        # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        self.h = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

