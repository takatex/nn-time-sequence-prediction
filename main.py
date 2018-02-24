import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
import time
import argparse
from utils import *
from visualizer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models.rnn import RNN
from models.lstm import LSTM
from models.qrnn import QRNN
from models.cnn import CNN

# params
# ----------
desc = 'time-series analysis using NN'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'qrnn', 'cnn'],
                    help='The type of model (default: rnn)')
parser.add_argument('--epoch', type=int, default=300,
                    help='The number of epochs to run (default: 300)')
parser.add_argument('--batch_size', type=int, default=200,
                    help='The number of batch (default: 200)')
parser.add_argument('--n_iter', type=int, default=5,
                    help='The number of iteration (default: 5)')
parser.add_argument('--seq_len', type=int, default=50,
                    help='The length of sequence (default: 50)')
parser.add_argument('--hidden_size', type=int, default=20,
                    help='The number of features in the hidden state h (default: 20)')
parser.add_argument('--num_layers', type=int, default=2,
                    help='The number of layers (default: 2)')
parser.add_argument('--result_path', type=str, default='./result',
                    help='Result path (default: ./result)')
parser.add_argument('--cuda', type=str, default='None',
                    help='set CUDA_VISIBLE_DEVICES (default: None)')

opt = parser.parse_args()
opt.result_path = os.path.join(opt.result_path, opt.model)
os.makedirs(opt.result_path, exist_ok=True)
if opt.cuda != 'None':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.cuda = True
else:
    opt.cuda = False


def models():
    if opt.model == 'rnn':
        return RNN(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif opt.model == 'lstm':
        return LSTM(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif opt.model == 'qrnn':
        return QRNN(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif opt.model == 'cnn':
        return CNN(1, opt.hidden_size, 1)

def train(rawdata, i):
    datasets = DATASETS(opt.seq_len, opt.batch_size, 1, opt.model)
    X_train, y_train, X_test, y_test = datasets.make(rawdata)
    model = models()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    if opt.cuda:
        model.cuda()
        criterion.cuda()

    # train
    # ----------
    loss_history = []
    time_history = []
    print("Train")
    for e in range(opt.epoch):
        total_loss = 0
        start = time.time()
        for iter_ in range(opt.n_iter):
            x, y = datasets.mini_traindata(X_train, y_train)
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            if opt.cuda:
                x = x.cuda()
                y = y.cuda()
            model.zero_grad()
            # model.reset()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()
            show_progress(e, iter_, opt.n_iter, loss.data[0])

        time_ = time.time() - start
        loss_history.append(total_loss/opt.n_iter)
        time_history.append(time_)

    # test
    # ----------
    print('\nTest')
    X_train, X_test = datasets.make_testdata(X_train, X_test)
    X_train = Variable(torch.from_numpy(X_train))
    X_test = Variable(torch.from_numpy(X_test))
    if opt.cuda:
        X_train = X_train.cuda()
        X_test = X_test.cuda()

    y_train_pred = model(X_train).cpu().data.numpy().reshape(-1)
    y_test_pred = model(X_test).cpu().data.numpy().reshape(-1)

    train_error, test_error = mse(y_train, y_train_pred, y_test, y_test_pred)
    figname = 'try_' + str(i) + '.png'
    plot_test(y_test, y_test_pred, show=False, save=True, save_path=os.path.join(opt.result_path, figname))

    return loss_history, time_history, train_error, test_error


def main():
    with open('./data/data.pkl', 'rb') as f:
        data = pickle.load(f)

    loss_history = []
    time_history = []
    train_error = []
    test_error = []
    for i in range(10):
        rawdata = data[:, i]
        # rawdata = data
        i_loss_history, i_time_history, i_train_error, i_test_error = train(rawdata, i)
        loss_history.append(i_loss_history)
        time_history.append(i_time_history)
        train_error.append(i_train_error)
        test_error.append(i_test_error)

    with open(os.path.join(opt.result_path, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(loss_history, f)
    with open(os.path.join(opt.result_path, 'time_history.pkl'), 'wb') as f:
        time_history = np.mean(time_history, axis=1).tolist()
        pickle.dump(time_history, f)
    with open(os.path.join(opt.result_path, 'train_error.pkl'), 'wb') as f:
        pickle.dump(train_error, f)
    with open(os.path.join(opt.result_path, 'test_error.pkl'), 'wb') as f:
        pickle.dump(test_error, f)


if __name__ == '__main__':
    main()
