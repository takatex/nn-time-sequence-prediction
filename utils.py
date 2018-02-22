# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import train_test_split




class DATASETS:
    def __init__(self, seq_len, batch_size, input_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.input_size = input_size

    def make(self, rawdata):
        test_size = 0.3
        X, y = [], []
        for i in range(len(rawdata)-self.seq_len):
            X.append(rawdata[i:i+self.seq_len])
            # y.append(rawdata[i+1:i+self.seq_len+1])
            y.append(rawdata[i+self.seq_len])
        X = np.array(X, dtype="float32")
        y = np.array(y, dtype="float32")
        (X_train, X_test, y_train, y_test) = \
                train_test_split(X, y, test_size=test_size, shuffle=False)
        self.N_train = len(X_train)
        N_test = len(X_test)
        return X_train, X_test, y_train, y_test, self.N_train, N_test


    def mini_traindata(self, X_train, y_train):
        X_train_mini, y_train_mini = [], []
        for i in range(self.batch_size):
            index = np.random.randint(0, self.N_train)
            X_train_mini.append(X_train[index])
            y_train_mini.append(y_train[index])
        # X_train_mini = np.array(X_train_mini, dtype="float32").reshape(self.seq_len, self.batch_size, self.input_size)
        # X_train_mini = np.array(X_train_mini, dtype="float32").reshape(self.batch_size,self.seq_len, self.input_size)
        X_train_mini = np.array(X_train_mini, dtype="float32").reshape(self.batch_size, self.input_size, self.seq_len)
        y_train_mini = np.array(y_train_mini, dtype="float32")

        return X_train_mini, y_train_mini
