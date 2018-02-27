# -*- coding: utf-8 -*-
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
parser.add_argument('--model', type=str, default='all', choices=['all', 'rnn', 'lstm', 'qrnn', 'cnn'],
                    help='The type of model (default: all)')
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
if opt.cuda != 'None':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.cuda = True
    print('Using GPU')
else:
    opt.cuda = False
    print('Using CPU')


def models(m):
    if m == 'rnn':
        return RNN(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif m == 'lstm':
        return LSTM(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif m == 'qrnn':
        return QRNN(1, opt.hidden_size, opt.num_layers, 1, opt.cuda)
    elif m == 'cnn':
        return CNN(1, opt.hidden_size, 1)

def train(i_data, m, i, result_path):
    datasets = DATASETS(opt.seq_len, opt.batch_size, 1)
    X_train, y_train, X_test, y_test = datasets.make(i_data)
    model = models(m)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    if opt.cuda:
        model.cuda()
        criterion.cuda()

    # train
    # ----------
    loss_history = []
    time_history = []
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
    X_train, X_test = datasets.make_testdata(X_train, X_test)
    X_train = Variable(torch.from_numpy(X_train))
    X_test = Variable(torch.from_numpy(X_test))
    if opt.cuda:
        X_train = X_train.cuda()
        X_test = X_test.cuda()

    y_train_pred = model(X_train).cpu().data.numpy().reshape(-1)
    y_test_pred = model(X_test).cpu().data.numpy().reshape(-1)

    train_error, test_error = mse(y_train, y_train_pred, y_test, y_test_pred)
    plot_test(i, y_test, y_test_pred, show=False, save=True, save_path=result_path)

    return loss_history, time_history, train_error, test_error


def main():
    if opt.model == 'all':
        ms = ['rnn', 'lstm', 'cnn', 'qrnn']
    else:
        ms = [opt.model]

    print('Fowllowing models will be used.', ms)
    for m in ms:
        print('\n\n**********************\n%s'%(m.upper()))
        print('**********************')
        result_path = os.path.join(opt.result_path, m)
        os.makedirs(result_path, exist_ok=True)

        with open('./data/data.pkl', 'rb') as f:
            data = pickle.load(f)

        loss_history = []
        time_history = []
        train_error = []
        test_error = []
        for i in range(10):
            print('\nmodel: %s - data %d/10' % (m, i+1))
            i_data = data[i]
            i_loss_history, i_time_history, i_train_error, i_test_error = train(i_data, m, i, result_path)
            loss_history.append(i_loss_history)
            time_history.append(i_time_history)
            train_error.append(i_train_error)
            test_error.append(i_test_error)

        with open(os.path.join(result_path, 'loss_history.pkl'), 'wb') as f:
            pickle.dump(loss_history, f)
        with open(os.path.join(result_path, 'time_history.pkl'), 'wb') as f:
            time_history = np.mean(time_history, axis=1).tolist()
            pickle.dump(time_history, f)
        with open(os.path.join(result_path, 'train_error.pkl'), 'wb') as f:
            pickle.dump(train_error, f)
        with open(os.path.join(result_path, 'test_error.pkl'), 'wb') as f:
            pickle.dump(test_error, f)

        plot_loss_history(m, save=True, save_path=result_path)

    try:
        error_boxplot(save=True, save_path=opt.result_path)
        time_boxplot(save=True, save_path=opt.result_path)
    except:
        message = '\n\nCannot draw a error boxplot and time boxplot.\nPossibly, there are not enough result files (*.pkl).'
        message += '\nIf you want to draw them, please run the following commands.\n\npython main.py --model all [other option]\n'
        print(message)

if __name__ == '__main__':
    main()
