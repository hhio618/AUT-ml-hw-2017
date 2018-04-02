# GaussianNaiveBayes implemented from Scratch in Python
#
# Created by Homayoun Heidarzadeh
#
import numpy as np
import math


def uni_gaussian(x, m, s):
    if s == 0.0:
        s = 0.00000001
    return (1.0 / (math.sqrt(2 * math.pi) * s)) * math.exp(-0.5 * (float(x - m) / s) ** 2)


class GaussianNaiveBayes:
    def __init__(self, features):
        self.params = None
        self.features = features

    def fit(self, train):
        num_data = train.shape[0]
        params = {}
        self.classes = np.unique(train[:, -1])
        for c in self.classes:
            idx = train[:, -1] == c
            train_c = train[idx, :-1]

            # add prior to class parameters
            params_c = [len(train_c) / float(num_data)]
            for feature in xrange(self.features):
                mu = np.mean(train_c[:, feature], axis=0)
                sigma = np.std(train_c[:, feature], axis=0)
                params_c.append((mu, sigma))
            params[c] = params_c
        self.params = params

    def predict(self, test_data, score=False):
        y_pred = []
        for row in test_data:
            probs = []
            for c in sorted(self.classes):
                params = self.params[c]
                # use prior (index at 0)
                prob = math.log(params[0])
                for i in xrange(self.features):
                    lh = uni_gaussian(row[i], *params[i + 1])
                    if lh > 0:
                        prob += math.log(lh)
                probs.append(prob)
            if not score:
                idx = probs.index(max(probs))
                result = self.classes[idx]
            else:
                result = probs[1] - probs[0]
            y_pred.append(result)
        return y_pred
