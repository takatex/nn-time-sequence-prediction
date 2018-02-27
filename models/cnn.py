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
            nn.Conv1d(input_size, hidden_size, 5, 3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(hidden_size, hidden_size, 4, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
            )
        self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
                )

    def forward(self, x):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(0, 1).transpose(1, 2)
        out = self.features(x)
        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        out = out.transpose(1, 2).transpose(0, 1)
        out = self.fc(out)
        return out
