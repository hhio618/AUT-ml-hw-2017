import numpy as np


def confusion_matrix(expected, predicted):
    classes = list(sorted(np.unique(expected)))[::-1]
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))
    for idx in xrange(len(expected)):
        cm[classes.index(predicted[idx])][classes.index(expected[idx])] += 1
    return cm


def sens(cm):
    return float(cm[0][0]) / cm[:, 0].sum()


def spec(cm):
    return float(cm[1][1]) / cm[:, 1].sum()


def acc(cm):
    return cm.trace() / cm.sum()
