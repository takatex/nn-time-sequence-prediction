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

## params
# args
# ----------
desc = 'time-series analysis using NN'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'qrnn', 'cnn'],
                    help='The type of model (default: rnn)')
parser.add_argument('--epoch', type=int, default=300,
                    help='The number of epochs to run')
parser.add_argument('--cuda', type=str, default='None',
                    help='set CUDA_VISIBLE_DEVICES (default: None)')
opt = parser.parse_args()
if opt.cuda != 'None':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
# others
# ----------
# DATA_PATH = './data/data.pkl'
DATA_PATH = './data/data2.pkl'
RESULT_PATH = os.path.join('./result', opt.model)
os.makedirs(RESULT_PATH, exist_ok=True)
SEQ_LEN = 50
INPUT_SIZE = 1 # The number of expected features in the input x
HIDDEN_SIZE = 10 # The number of features in the hidden state h
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 200
N_ITER = 5


def models():
    if opt.model == 'rnn':
        return RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    elif opt.model == 'lstm':
        return LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    elif opt.model == 'qrnn':
        return QRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, use_cuda=opt.gpu)
    elif opt.model == 'cnn':
        return CNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

def train(rawdata, i):
    datasets = DATASETS(SEQ_LEN, BATCH_SIZE, INPUT_SIZE, opt.model)
    X_train, y_train, X_test, y_test = datasets.make(rawdata)
    model = models()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    if opt.cuda != 'None':
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
        for iter_ in range(N_ITER):
            x, y = datasets.mini_traindata(X_train, y_train)
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            if opt.cuda != 'None':
                x = x.cuda()
                y = y.cuda()
            model.zero_grad()
            # model.reset()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()
            show_progress(e, iter_, N_ITER, loss.data[0])

            # if (e+1) % 100 == 0:
            #     print("epoch:\t{}\t loss:\t{}".format(e+1, loss.data[0]))

        time_ = time.time() - start
        loss_history.append(total_loss/N_ITER)
        time_history.append(time_)

    # test
    # ----------
    print('\nTest')
    X_train, X_test = datasets.make_testdata(X_train, X_test)
    X_train = Variable(torch.from_numpy(X_train))
    X_test = Variable(torch.from_numpy(X_test))
    if opt.cuda != 'None':
        X_train = X_train.cuda()
        X_test = X_test.cuda()

    y_train_pred = model(X_train).cpu().data.numpy().reshape(-1)
    y_test_pred = model(X_test).cpu().data.numpy().reshape(-1)

    train_error, test_error = mse(y_train, y_train_pred, y_test, y_test_pred)
    figname = 'try_' + str(i) + '.png'
    plot_test(y_test, y_test_pred, show=False, save=True, save_path=os.path.join(RESULT_PATH, figname))


    return loss_history, time_history, train_error, test_error


def main():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    loss_history = []
    time_history = []
    train_error = []
    test_error = []
    for i in range(10):
        # rawdata = data[:, i]
        rawdata = data
        i_loss_history, i_time_history, i_train_error, i_test_error = train(rawdata, i)
        loss_history.append(i_loss_history)
        time_history.append(i_time_history)
        train_error.append(i_train_error)
        test_error.append(i_test_error)

    with open(os.path.join(RESULT_PATH, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(loss_history, f)
    with open(os.path.join(RESULT_PATH, 'time_history.pkl'), 'wb') as f:
        pickle.dump(time_history, f)
    with open(os.path.join(RESULT_PATH, 'train_error.pkl'), 'wb') as f:
        pickle.dump(train_error, f)
    with open(os.path.join(RESULT_PATH, 'test_error.pkl'), 'wb') as f:
        pickle.dump(test_error, f)


if __name__ == '__main__':
    main()
