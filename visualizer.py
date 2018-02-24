# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


def show_save(show, save, save_path):
    if save:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test(y_test, y_test_pred, show=False, save=False, save_path=''):
    plt.plot(y_test, color='blue')
    plt.plot(y_test_pred, color='red')
    show_save(show, save, save_path)


def error_boxplot(rnn_error, lstm_error, cnn_error, qrnn_error, show=False, save=False, save_path=''):
    error = rnn_error + lstm_error + cnn_error + qrnn_error
    len_ = len(error)
    iter_ = int(len_/8)
    model_type = ['rnn']* iter_ * 2 + ['lstm'] * iter_ * 2 + ['cnn'] * iter_*2 + ['qrnn'] * iter_*2
    error_type = (['train'] * iter_ + ['test'] * iter_) * 4

    data =  pd.DataFrame({'model_type' : model_type, 'error_type' : error_type, 'error' : error})
    ax = sns.boxplot(x="model_type", y="error", hue="error_type", data=data)
    show_save(show, save, save_path)


def time_boxplot(rnn_time, lstm_time, cnn_time, qrnn_time, show=False, save=False, save_path=''):
    time_ = rnn_time + lstm_time + cnn_time + qrnn_time
    len_ = len(time_)
    print(len_)
    iter_ = int(len_/4)
    model_type = ['rnn']* iter_ + ['lstm'] * iter_ + ['cnn'] * iter_ + ['qrnn'] * iter_

    data =  pd.DataFrame({'model_type' : model_type, 'time' : time_})
    ax = sns.boxplot(x="model_type", y='time', data=data)
    show_save(show, save, save_path)
