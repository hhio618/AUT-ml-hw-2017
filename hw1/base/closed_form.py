import numpy as np


class ClosedFormRegression(object):
    def __init__(self, order, lambda_var):
        self.order = order
        self.lambda_var = lambda_var

    def fit(self, X_train, y_train):
        H = self.calculate_H(X_train)
        # l2 regularization
        G = np.eye(H.shape[1])
        # ignore theta[0]
        G[0, 0] = 0
        self.theta = np.linalg.inv(H.T.dot(H) + self.lambda_var * G).dot(H.T).dot(y_train)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.sum((y - y_pred) ** 2) / (2 * y_pred.shape[0])

    def calculate_H(self, X):
        H = np.c_[np.ones((X.shape[0], 1)), X]
        for i in range(2, self.order + 1):
            H = np.c_[H, np.power(H[:, 1], i)]
        return H

    def predict(self, X_unseen):
        H_unseen = self.calculate_H(X_unseen)
        return H_unseen.dot(self.theta)

    def description(self):
        return "ord=%d, lambda=%d" % (self.order, self.lambda_var)
