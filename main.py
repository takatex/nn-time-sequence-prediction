import pickle
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse

from utils import *
from models import *

## params
# args
# ----------
desc = 'time-series analysis using NN'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'tdnn', 'qrnn'],
                    help='The type of model (default: rnn)')
parser.add_argument('--epoch', type=int, default=300,
                    help='The number of epochs to run')
parser.add_argument('--gpu', type=str, default='None',
                    help='set CUDA_VISIBLE_DEVICES (default: None)')
opt = parser.parse_args()
# others
# ----------
DATA_PATH = './data/data.pkl'
RESULT_PATH = './result'
EPOCH_NUM = 300
INPUT_SIZE = 10000
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1
BATCH_SIZE = 100


def train(rawdata):
    datasets = DATASETS(INPUT_SIZE, BATCH_SIZE)
    X_train, X_test, y_train, y_test, N_train, N_test = datasets.make(rawdata)
    model = models(opt.model, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
   
    # train
    # ----------
    print("Train")
    for e in range(opt.epoch):
        total_loss = 0
        for j in range(5):
            x, y = datasets.mini_traindata(X_train, y_train)
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            # model.reset()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()

        if (e+1) % 10 == 0:
            print("epoch:\t{}\t loss:\t{}".format(e+1, loss.data[0]))

    # test
    # ----------
    print('\nTest')
    # model.reset()
    # for i in range(N_test):
    # x = Variable(torch.from_numpy(X_train.reshape(1, N_train, INPUT_SIZE)))
    # x = Variable(torch.from_numpy(X_train.T.reshape(INPUT_SIZE, N_train, 1)))
    # y_pred_1 = model(x, True).data.numpy().reshape(-1)
    # y = y_pred_1
    # model.reset()
    # x = Variable(torch.from_numpy(X_test.reshape(1, N_test, INPUT_SIZE)))
    x = Variable(torch.from_numpy(X_test.T.reshape(INPUT_SIZE, N_test, 1)))
    y_pred_2 = model(x).data.numpy().reshape(-1)

    y = y_pred_2
    # y = np.r_[y_pred_1, y_pred_2]    
    print(rawdata.shape)
    print(y.shape)
    print('unko')
    plt.plot(rawdata, color='blue')
    plt.plot(y, color='r')
    plt.savefig('./test.png')
    plt.close()









def main():
    if opt.gpu != 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    for i in range(10):
        rawdata = data[:, i]
        train(rawdata)




if __name__ == '__main__':
    main()
