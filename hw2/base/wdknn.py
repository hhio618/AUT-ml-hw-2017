# WDKNN implemented from Scratch in Python
#
# Created by Homayoun Heidarzadeh
#
import numpy as np
import math
from base.distances import EuclideanRowWise
from base.knn import KNN


class WDKNN(KNN):
    def __init__(self, train_data, train_label):
        KNN.__init__(self, train_data, train_label)
        self.major = int(train_label[train_label == 1].shape[0] > train_label[train_label == 0].shape[0])
        self.minor = 1 - self.major

    def predict(self, x, k, row_wise_distance_func):
        # find index of k nearest neighbor
        distances = EuclideanRowWise()(x, self.train_data)
        idx = np.argpartition(distances, k)
        # we want first k -->
        distances_idx = idx[:k]
        k_distances = distances[distances_idx]
        # find corresponding neighbor labels
        neighbor_classes = self.train_label[distances_idx]
        idx_maj = (neighbor_classes == self.major)
        idx_min = (neighbor_classes == self.minor)
        I_maj = np.sum(1. / k_distances[idx_maj])
        I_min = np.sum(1. / k_distances[idx_min])
        # return most frequent neighbor class
        if np.isinf(I_maj):
            return 0
        if np.isinf(I_min):
            return 1
        return self.major if I_maj > I_min else self.minor

    def evaluate(self, validation_set, validation_label, k, row_wise_distance_func):
        count = validation_set.shape[0]
        (tp, tn, fp, fn) = (0, 0, 0, 0)
        # Calculate tp, tn, fp, fn per test sample
        for idx in range(count):
            predicted = self.predict(validation_set[idx], k, row_wise_distance_func)
            label = validation_label[idx]
            if predicted == 1:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if label == 1:
                    fn += 1
                else:
                    tn += 1
        # Calculate precision, recall, etc.
        precision = 1.0 if tp + fp == 0 else float(tp) / (tp + fp)
        recall = 1.0 if tp + fn == 0 else float(tp) / (tp + fn)
        f1 = (2 * recall * precision) / (recall + precision)
        g_mean = math.sqrt((tp / float(tp + fn)) * (tn / float(tn + fp)))
        return [precision, recall, f1, g_mean]

    @classmethod
    def cross_validation_for_k(cls, k_list, data, label):
        results = []
        for k in k_list:
            # Append evaluated results
            result = cls.k_fold(5, k, EuclideanRowWise(), data, label)
            results.append(result)
            # print "k =", k, "avg_acc =", avg_acc
        return np.mean(results, axis=0)

    @classmethod
    def k_fold(cls, folds_count, k, row_wise_distance_func, data, label):
        results = []
        for i in range(folds_count):
            trains, train_labels, validations, validation_label = cls.fold_data(i, folds_count, data, label)
            # Evaluate this fold
            result = cls(trains, train_labels).evaluate(validations, validation_label, k, row_wise_distance_func)
            results.append(result)
        return np.mean(results, 0)
