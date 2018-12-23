# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:17:08 2018

@author: jasap
"""

import pandas as pd
import numpy as np

#np.random.seed(1)
#np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})

data = pd.read_csv('Boston_Housing.txt',
                   sep='\t')

def prepare(data):
    y = data.iloc[:, [-1]]
    X = data.iloc[:, 0:-1]

    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std

    X['X0'] = 1

    y = y.as_matrix()
    X = X.as_matrix()

    n, m = X.shape
    w = np.random.random((1, m))
    # alpha = 0.06
    # lam = 1
    return X, w, y, n


def learn(data, alpha=0.06, lam=1):
    X, w, y, n = prepare(data)
    for iter in range(10000):
        pred = X.dot(w.T)
        err = pred-y
        reg = 2 * lam * w
        reg[0][-1] = 0
        grad = (err.T.dot(X) / n) + reg
        w = w - alpha * grad
        mean_square_error = err.T.dot(err) / n
        grad_norm = abs(grad).sum()
        if grad_norm < 0.1 or mean_square_error < 10: break
        print(iter, grad_norm, mean_square_error,reg.sum())
    return w


# URADJEN online dodat shuffle

def online_learn(data, alpha=0.06):
    X, w, y, n = prepare(data)
    for iter in range(10):
        print('Epoch:', iter)
        np.random.shuffle(X)
        for i, row in enumerate(X):
            pred = row.dot(w.T)
            err = pred - y[i]
            grad = err * row
            w = w - alpha * grad
        pred = X.dot(w.T)
        err = pred - y
        grad = err.T.dot(X) / n
        mean_square_error = err.T.dot(err) / n
        print(mean_square_error)


learn(data,lam=10)
online_learn(data)


# data_new = pd.read_csv('C:\\Users\\jasap\\Documents\\MASTER\\AlgMasUc\\2. Linear regression\\Kod\\house_new.csv')
# data_new = (data_new-X_mean)/X_std
# data_new['X0'] = 1
# prediction = data_new.as_matrix().dot(w.T)
# print(prediction)
