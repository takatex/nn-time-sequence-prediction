import numpy as np
from rnn import RNN
from lstm import LSTM
from qrnn import QRNN
from cnn import CNN


def models(model_type, input_size, hidden_size, num_layers, output_size):
    if model_type == 'rnn':
        model = RNN(input_size, hidden_size, num_layers, output_size)
    elif model_type == 'lstm':
        model = LSTM(input_size, hidden_size, num_layers, output_size)
    elif model_type == 'qrnn':
        model = QRNN(input_size, hidden_size, num_layers, output_size, use_cuda=False)
    elif model_type == 'cnn':
        model = CNN(input_size, hidden_size, output_size)

    return model

