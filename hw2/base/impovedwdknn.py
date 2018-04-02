# ImprovedWDKNN implemented from Scratch in Python
#
# Created by Homayoun Heidarzadeh
#
import numpy as np
from base.distances import EuclideanRowWise
from base.wdknn import WDKNN


class ImprovedWDKNN(WDKNN):
    def __init__(self, train_data, train_label):
        WDKNN.__init__(self, train_data, train_label)

    def predict(self, x, k, row_wise_distance_func):
        # find index of k nearest neighbor
        distances = EuclideanRowWise()(x, self.train_data)
        idx = np.argpartition(distances, k)
        # we want first k -->
        distances_idx = idx[:k]
        k_distances = distances[distances_idx]
        # find corresponding neighbor labels
        neighbor_classes = self.train_label[distances_idx]
        I_min = 0
        for index in distances_idx:
            if self.train_label[index] == self.minor:
                x1 = self.train_data[index]
                distances1 = EuclideanRowWise()(x1, self.train_data)
                index1 = np.argpartition(distances1, k)
                # we want first k -->
                distances_idx1 = index1[:k]
                minor_neighbor_classes = self.train_label[distances_idx1]
                N_maj = minor_neighbor_classes[minor_neighbor_classes == self.major].shape[0]
                L_min = N_maj / float(k)
                I_min += (L_min + 1.) / distances[index]
        idx_maj = (neighbor_classes == self.major)
        I_maj = np.sum(1. / k_distances[idx_maj])
        # return most frequent neighbor class
        if np.isinf(I_maj):
            return 0
        if np.isinf(I_min):
            return 1
        return self.major if I_maj > I_min else self.minor
