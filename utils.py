# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class DATASETS:
    def __init__(self, seq_len, batch_size, input_size, model_type):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.model_type = model_type

    def make(self, rawdata):
        rawdata_len = len(rawdata) - self.seq_len
        train_size = 0.7
        self.train_len = int(rawdata_len * train_size)
        self.test_len = rawdata_len - self.train_len

        X_train = np.array([rawdata[i:i+self.seq_len]
                for i in range(self.train_len)], dtype="float32")
        y_train = np.array([rawdata[i+self.seq_len]
                for i in range(self.train_len)], dtype="float32")

        X_test = np.array([rawdata[i:i+self.seq_len]
                for i in range(self.train_len, rawdata_len)], dtype="float32")
        y_test = np.array([rawdata[i+self.seq_len]
                for i in range(self.train_len, rawdata_len)], dtype="float32")

        return X_train, y_train, X_test, y_test


    def mini_traindata(self, X_train, y_train):
        X_train_mini, y_train_mini = [], []
        for i in range(self.batch_size):
            index = np.random.randint(0, self.train_len)
            X_train_mini.append(X_train[index])
            y_train_mini.append(y_train[index])
        X_train_mini = np.array(X_train_mini, dtype="float32")
        X_train_mini = X_train_mini.T.reshape(self.seq_len, self.batch_size, self.input_size)
        # X_train_mini = X_train_mini.T.reshape(self.batch_size, self.seq_len)
        # X_train_mini = np.array(X_train_mini, dtype="float32").reshape(self.batch_size, self.input_size, self.seq_len)
        y_train_mini = np.array(y_train_mini, dtype="float32")

        return X_train_mini, y_train_mini

    def make_testdata(self, X_train, X_test):
        X_train = X_train.T.reshape(self.seq_len, self.train_len, self.input_size)
        X_test = X_test.T.reshape(self.seq_len, self.test_len, self.input_size)

        return X_train, X_test


def mse(y_train, y_train_pred, y_test, y_test_pred):
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    return train_error, test_error

def show_progress(e,b,b_total,loss):
    sys.stdout.write("\r%3d: [%5d / %5d] loss: %f" % (e, b, b_total, loss))
    sys.stdout.flush()
