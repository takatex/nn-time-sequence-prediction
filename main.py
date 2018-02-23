import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from utils import *

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
parser.add_argument('--gpu', type=str, default='None',
                    help='set CUDA_VISIBLE_DEVICES (default: None)')
opt = parser.parse_args()
# others
# ----------
DATA_PATH = './data/data.pkl'
# DATA_PATH = './data/data2.pkl'
RESULT_PATH = './result'
os.makedirs(RESULT_PATH, exist_ok=True)
SEQ_LEN = 10
INPUT_SIZE = 1 # The number of expected features in the input x
HIDDEN_SIZE = 5 # The number of features in the hidden state h
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 200
N_ITER = 1


def models():
    if opt.model == 'rnn':
        return RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    elif opt.model == 'lstm':
        return LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    elif opt.model == 'qrnn':
        return QRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, use_cuda=opt.gpu)
    elif opt.model == 'cnn':
        return CNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

def train(rawdata, data_num):
    datasets = DATASETS(SEQ_LEN, BATCH_SIZE, INPUT_SIZE, opt.model)
    X_train, y_train, X_test, y_test = datasets.make(rawdata)
    print(X_train.shape)
    print(X_test.shape)
    model = models()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
   
    # train
    # ----------
    print("Train")
    for e in range(opt.epoch):
        total_loss = 0
        for iter_ in range(N_ITER):
            x, y = datasets.mini_traindata(X_train, y_train)
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            # model.reset()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()

        if (e+1) % 100 == 0:
            print("epoch:\t{}\t loss:\t{}".format(e+1, loss.data[0]))

    # test
    # ----------
    print('\nTest')
    X_train, y_train, X_test, y_test = \
            datasets.make_testdata(X_train, y_train, X_test, y_test)
    X_train = Variable(torch.from_numpy(X_train))
    X_test = Variable(torch.from_numpy(X_test))
    y_train_pred = model(X_train).data.numpy()
    y_test_pred = model(X_test).data.numpy()
    if opt.model == 'cnn':
        y_train_pred = y_train_pred.reshape(-1)
        y_test_pred = y_test_pred.reshape(-1)
    else:
        y_train_pred = y_train_pred[SEQ_LEN-1, :, :].reshape(-1)
        y_test_pred = y_test_pred[SEQ_LEN-1, :, :].reshape(-1)
    train_error, test_error = mse(y_train, y_train_pred, y_test, y_test_pred)
    plt.plot(y_test, color='blue')
    plt.plot(y_test_pred, color='red')
    figname = 'data' + str(data_num) + '.png'
    plt.savefig(os.path.join(RESULT_PATH, figname))
    plt.close()

def main():
    if opt.gpu != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    # rawdata = data
    # train(rawdata, 100)
    for i in range(10):
        rawdata = data[:, i]
        train(rawdata, i)

if __name__ == '__main__':
    main()
