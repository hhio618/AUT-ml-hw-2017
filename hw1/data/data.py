import pandas
import numpy as np


def _normalize(X, mean, std):
    return (X - mean) / std


def load(train_coefficient, normalize):
    df = pandas.read_excel('data/data.xlsx')
    data = df.values
    # shuffle data
    np.random.shuffle(data)
    # slice data to train and test
    idx = int(train_coefficient * data.shape[0])
    X_train, y_train = np.atleast_2d(data[idx:, 0]).T, data[idx:, 1]
    X_test, y_test = np.atleast_2d(data[:idx, 0]).T, data[:idx, 1]
    # normalize data
    if normalize:
        train_mean = np.mean(X_train, 0)
        train_std = np.std(X_train, 0)
        X_train = _normalize(X_train, train_mean, train_std)
        X_test = _normalize(X_test, train_mean, train_std)
    return X_train, y_train, X_test, y_test
