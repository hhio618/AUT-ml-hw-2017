import numpy as np
import math


class GradientDescentRegression(object):
    def __init__(self, order, alpha):
        # variable initializing
        self.order = order
        self.theta = None
        self.num_iteration = None
        self.alpha = alpha

    def fit(self, X_train, y_train, X_test, y_test, num_iteration):
        self.num_iteration = num_iteration
        m = X_train.shape[0]
        # calculated feature
        H = self.calculate_H(X_train)
        xTrans = H.T
        # theta initializing
        self.theta = np.random.normal(size=H.shape[1])
        mse_train, mse_test = [], []
        mod = int(math.log(num_iteration))
        for i in range(0, num_iteration):
            hypothesis = np.dot(H, self.theta)
            loss = hypothesis - y_train
            # avg cost per example
            cost = np.sum(loss ** 2) / (2 * m)
            # avg gradient per example
            gradient = np.dot(xTrans, loss) / m
            if i % mod == 0 or i == num_iteration - 1:
                mse_train.append(cost)
                mse_test.append(self.mse(X_test, y_test))
            # update
            self.theta = self.theta - self.alpha * gradient

        return mse_train, mse_test

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
        return "ord=%d, alpha=%g, iter=%d" % (self.order, self.alpha, self.num_iteration)
