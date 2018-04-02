# KNN implemented from Scratch in Python
#
# Created by Homayoun Heidarzadeh
#
import numpy as np

from base.distances import EuclideanRowWise


class KNN:
    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def evaluate(self, vaidation_set, validation_label, k, row_wise_distance_func):
        error = 0
        count = vaidation_set.shape[0]
        # Calculate error per test sample
        for idx in range(count):
            predicted = self.predict(vaidation_set[idx], k, row_wise_distance_func)
            label = validation_label[idx]
            if predicted != label:
                error += 1. / count
        return error

    def predict(self, x, k, row_wise_distance_func):
        # find index of k nearest neighbor
        idx = np.argpartition(row_wise_distance_func(x, self.train_data), k)
        # we want first k -->
        distances_idx = idx[:k]
        # find corresponding neighbor labels
        neighbor_classes = self.train_label[distances_idx]
        # return most frequent neighbor class
        return np.argmax(np.bincount(neighbor_classes))

    @classmethod
    def cross_validation_for_k(cls, k_list, data, label):
        avg_accs = []
        for k in k_list:
            avg_acc = 1 - cls.k_fold(10, k, EuclideanRowWise(), data, label)
            avg_accs.append(avg_acc)
            # print "k =", k, "avg_acc =", avg_acc
        return np.c_[k_list, avg_accs]

    @staticmethod
    def cross_validation_for_distance(k, list_row_wise_distance_func, data, label):
        avg_accs = []
        for row_wise_distance_func in list_row_wise_distance_func:
            # Get avg accuracy
            avg_acc = 1 - KNN.k_fold(10, k, row_wise_distance_func, data, label)
            avg_accs.append(avg_acc)
            # print "k =", k, "avg_acc =", avg_acc
        return np.c_[list_row_wise_distance_func, avg_accs]

    @classmethod
    def k_fold(cls, folds_count, k, row_wise_distance_func, data, label):
        errors = []
        for i in range(folds_count):
            # Get new fold
            trains, train_labels, validations, validation_label = cls.fold_data(i, folds_count, data, label)
            # Evaluate this fold
            error = cls(trains, train_labels).evaluate(validations, validation_label, k, row_wise_distance_func)
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
