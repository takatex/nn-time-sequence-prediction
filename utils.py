# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import train_test_split




class DATASETS:
    def __init__(self, input_size, batch_size):
        self.input_size = input_size
        self.batch_size = batch_size

    def make(self, rawdata):
        test_size = 0.3
        X, y = [], []
        for i in range(len(rawdata)-self.input_size):
            X.append(rawdata[i:i+self.input_size])
            # y.append(rawdata[i+1:i+self.input_size+1])
            y.append(rawdata[i+self.input_size])
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
        # X_train_mini = np.array(X_train_mini, dtype="float32").reshape(1, self.batch_size, self.input_size)
        X_train_mini = np.array(X_train_mini, dtype="float32").reshape(self.input_size, self.batch_size,1)
        y_train_mini = np.array(y_train_mini, dtype="float32")
        # print('y_trai_mini', y_train_mini.shape)

        return X_train_mini, y_train_mini
