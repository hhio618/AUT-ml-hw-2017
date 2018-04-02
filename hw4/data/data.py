import numpy as np


def load_data1():
    np.random.seed(5)
    train_data = np.genfromtxt("data/data1.csv", dtype=float, delimiter=',', skip_header=1)
    np.random.shuffle(train_data)
    X = train_data[:, :-1]
    y = train_data[:, -1]
    return X, y.astype('int')


def normalize(X):
    return (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def load_parkinson():
    np.random.seed(0)
    idx_features = range(23)
    del idx_features[16]
    data = np.genfromtxt("data/parkinsons.data", dtype=float, delimiter=',', skip_header=1, usecols=xrange(1, 24))
    np.random.shuffle(data)
    data[:, idx_features] = normalize(data[:, idx_features])

    m = len(data)
    train_idx = int(.7 * m)
    validation_idx = int(.85 * m)
    train_data = data[:train_idx]
    validation_data = data[train_idx:validation_idx]
    test_data = data[validation_idx:]
    X_tr = train_data[:, idx_features]
    y_tr = train_data[:, 16].astype('int')
    X_va = validation_data[:, idx_features]
    y_va = validation_data[:, 16].astype('int')
    X_te = test_data[:, idx_features]
    y_te = test_data[:, 16].astype('int')
    return X_tr, y_tr, X_va, y_va, X_te, y_te
