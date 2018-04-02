import numpy as np
import pandas as pd


def load_data1():
    np.random.seed(5)
    train_data = pd.read_excel("data/data1.xlsx", dtype=float).values
    X0 = train_data[:, 0]
    Y1 = train_data[:, 1]
    Y2 = train_data[:, 2]
    X = np.r_[np.c_[X0, Y1], np.c_[X0, Y2]]
    return normalize(X)


def load_data2():
    X = np.loadtxt('data/data2.csv', delimiter=',')
    return normalize(X)


def load_s1():
    X = np.loadtxt('data/s1.txt')
    return normalize(X)


def normalize(X):
    return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
