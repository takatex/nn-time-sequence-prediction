# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import pandas as pd
import pandas as pd


import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('poster')


def show_save(show, save, save_path):
    if save:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def load_loss_history(model):
    path = os.path.join('./result', model, 'loss_history.pkl')
    with open(path, 'rb') as f:
        loss_history = pickle.load(f)
    return loss_history


def load_error(model):
    train_error_path = os.path.join('./result', model, 'train_error.pkl')
    test_error_path = os.path.join('./result', model, 'test_error.pkl')
    with open(train_error_path, 'rb') as f:
        train_error = pickle.load(f)
    with open(test_error_path, 'rb') as f:
        test_error = pickle.load(f)
    error = train_error + test_error
    return error


def load_time_history(model):
    path = os.path.join('./result', model, 'time_history.pkl')
    with open(path, 'rb') as f:
        time_history = pickle.load(f)
    return time_history


def plot_test(i, y_test, y_test_pred, show=False, save=False, save_path=''):
    figname = 'data%s.png'%(i)
    save_path = os.path.join(save_path, figname)
    plt.plot(y_test[:300], color='blue')
    plt.plot(y_test_pred[:300], color='red')
    show_save(show, save, save_path)


def plot_loss_history(model, show=False, save=False, save_path=''):
    figname = 'loss_history.png'
    save_path = os.path.join(save_path, figname)
    loss_history = load_loss_history(model)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.array(loss_history).T)
    ax.set_title(model.upper())
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    show_save(show, save, save_path)


def error_boxplot(show=False, save=False, save_path=''):
    figname = 'error.png'
    save_path = os.path.join(save_path, figname)
    rnn_error = load_error('rnn')
    lstm_error = load_error('lstm')
    cnn_error = load_error('cnn')
    qrnn_error = load_error('qrnn')
    error = rnn_error + lstm_error + cnn_error + qrnn_error
    len_ = len(error)
    iter_ = int(len_/8)
    model = ['RNN']* iter_ * 2 + ['LSTM'] * iter_ * 2 + ['CNN'] * iter_*2 + ['QRNN'] * iter_*2
    error_type = (['train'] * iter_ + ['test'] * iter_) * 4

    data =  pd.DataFrame({'model' : model, 'type' : error_type, 'MSE' : error})
    ax = sns.boxplot(x='model', y='MSE', hue="type", data=data)
    show_save(show, save, save_path)


def time_boxplot(show=False, save=False, save_path=''):
    figname = 'time.png'
    save_path = os.path.join(save_path, figname)
    rnn_time = load_time_history('rnn')
    lstm_time = load_time_history('lstm')
    cnn_time = load_time_history('cnn')
    qrnn_time = load_time_history('qrnn')
    time_ = rnn_time + lstm_time + cnn_time + qrnn_time
    len_ = len(time_)
    iter_ = int(len_/4)
    model = ['RNN']* iter_ + ['LSTM'] * iter_ + ['CNN'] * iter_ + ['QRNN'] * iter_

    data =  pd.DataFrame({'model' : model, 'time [sec/epoch]' : time_})
    ax = sns.boxplot(x="model", y='time [sec/epoch]', data=data)
    show_save(show, save, save_path)
