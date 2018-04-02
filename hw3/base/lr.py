# GaussianNaiveBayes implemented from Scratch in Python
#
# Created by Homayoun Heidarzadeh
#

import matplotlib

from base.utils import confusion_matrix, acc

matplotlib.use('TKAgg')
matplotlib.interactive(False)
import numpy as np


def phi(t, dt=False):
    if dt:
        return phi(t) * (1. - phi(t))
    else:
        return 1. / (1. + np.exp(-t))


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.mean = None
        self.std = None

    def fit(self, X, y, num_iterations, learning_rate, lam, print_cost=False):
        self.normalize(X)
        X = np.c_[np.ones((len(X), 1)), X]
        W = np.random.uniform(-0.1, 0.1, X.shape[1])
        losses = []
        for i in range(num_iterations):
            dJw = self.propagate(X, y, W, lam)
            W -= learning_rate * dJw
            if i % 100 == 0:
                print "Iteration @ %d" % i
        self.W = W
        return self

    def propagate(self, X, y, W, lam):
        # calculate the prediction
        Z = np.dot(X, W)
        # temporary weight vector
        W1 = np.copy(W)  # import copy to create a true copy
        W1[0] = 0
        # calc gradient
        grad = (1. / len(X)) * (np.dot((phi(Z) - y).T, X).T) + (lam / len(X)) * W1
        return grad

    def predict(self, X, score=False):
        X = self.normalize(X)
        X = np.c_[np.ones((len(X), 1)), X]
        A = phi(np.dot(X, self.W))
        if score:
            return A
        else:
            y_prediction = 2. * (A > 0.5) - 1
            return y_prediction

    def evaluate(self, X):
        y_true = X[:, -1]
        y_pred = self.predict(X[:, :-1])
        cm = confusion_matrix(y_true, y_pred)
        return cm, float(cm[1][0] + cm[0][1]) / cm.sum()

    def evaluate_report(self, X):
        cm, _ = self.evaluate(X)
        print cm
        print acc(cm)

    def normalize(self, X):
        if self.mean is None and self.std is None:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        return (X - self.mean) / self.std

    @staticmethod
    def cross_validation_for_lambda(list_lambda, X, y):
        avg_errors = []
        for lam in list_lambda:
            # Get avg accuracy
            avg_error = LogisticRegression.k_fold(10, X, y, 2000, 0.01, 10. ** lam, False)
            avg_errors.append(avg_error)
        return avg_errors

    @classmethod
    def k_fold(cls, folds_count, X, y, num_iterations, learning_rate, lam, print_cost=False):
        errors = []
        for i in range(folds_count):
            # Get new fold
            trains, train_labels, validations, validation_label = cls.fold_data(i, folds_count, X, y)
            # Evaluate this fold
            _, error = cls().fit(trains, train_labels, num_iterations, learning_rate, lam, print_cost) \
                .evaluate(np.c_[validations, validation_label])
            errors.append(error)
        return np.mean(errors)

    @classmethod
    def fold_data(cls, i, folds_count, train_data, train_label):
        # Folding data
        unique_label = np.unique(train_label)
        trains = []
        train_labels = []
        validations = []
        validation_labels = []
        for label in unique_label:  # unique_label:
            idx = train_label == label
            class_train = train_data[idx]
            class_label = train_label[idx]
            data_count = class_train.shape[0]
            fold_size = data_count / folds_count
            fold_start, folds_end = i * fold_size, (i + 1) * fold_size

            validations.append(class_train[fold_start:folds_end])
            validation_labels.append(class_label[fold_start:folds_end])
            if fold_start != 0:
                trains.append(class_train[0:fold_start])
                train_labels.append(class_label[0:fold_start])
            if folds_end < data_count:
                trains.append(class_train[folds_end:])
                train_labels.append(class_label[folds_end:])

        out = np.vstack(trains), np.hstack(train_labels), np.vstack(validations), np.hstack(validation_labels)
        return out
