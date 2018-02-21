from rnn import RNN
from lstm import LSTM


def models(model_type, input_size, hidden_size, output_size):
    if model_type == 'rnn':
        model = RNN(input_size, hidden_size, output_size)
    elif model_type == 'lstm':
        model = LSTM(input_size, hidden_size, output_size)

    return model

