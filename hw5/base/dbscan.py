# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from matplotlib import cm
from matplotlib import mpl


class DBScan:
    # if X is big the we do sampling
    @staticmethod
    def hyper_tuner(plt, X):
        count = X.shape[0]
        min_distances = list()
        for i, xi in enumerate(X):
            min_i = sys.maxsize
            for j, xj in enumerate(X):
                if i != j:
                    min_i = min(min_i, np.linalg.norm(xj - xi))
            min_distances.append(min_i)
        plt.figure()
        freqs, bins, _ = plt.hist(min_distances, normed=False, bins=20)
        plt.savefig("outputs%smain1%sdbscan-tuner-eps.png" % (os.sep, os.sep))
        # find eps from hist
        eps = bins[-1]
        freq_sum = 0
        for i, freq in enumerate(freqs):
            if freq_sum >= 0.95 * count:
                eps = bins[i + 1]
                break
            freq_sum += freq

        eps_neighbourhood = list()
        for i, xi in enumerate(X):
            num_neighbour = 0
            for j, xj in enumerate(X):
                if i != j:
                    if np.linalg.norm(xj - xi) <= eps:
                        num_neighbour += 1
            eps_neighbourhood.append(num_neighbour)
        plt.figure()
        freqs, bins, _ = plt.hist(eps_neighbourhood, normed=False, bins=len(set(eps_neighbourhood)))
        plt.savefig("outputs%smain1%sdbscan-tuner-minpts.png" % (os.sep, os.sep))
        # find minPts from hist
        assert isinstance(freqs, np.ndarray)
        max_freqs_bin = freqs.argmax() + 1
        minPts = max_freqs_bin
        return eps, int(minPts)

    def __init__(self):
        self.X = None

    @staticmethod
    def eps_neighbours(X, P, eps):
        neighbors = []
        # For each point in the dataset...
        for Pn in range(0, len(X)):
            # If the distance is below the threshold, add it to the neighbors list.
            if np.linalg.norm(X[P] - X[Pn]) <= eps:
                neighbors.append(Pn)

        return neighbors

    def extend_cluster(self, X, labels, P, NeighborPts, C, eps, MinPts):

        # Assign the cluster label to the seed point.
        labels[P] = C
        i = 0
        while i < len(NeighborPts):
            # Get the next point from the queue.
            Pn = NeighborPts[i]
            # If Pn was labelled NOISE during the seed search, then we
            if labels[Pn] == -1:
                labels[Pn] = C

            # Otherwise, if Pn isn't already claimed, claim it as part of C.
            elif labels[Pn] == 0:
                # Add Pn to cluster C (Assign cluster label C).
                labels[Pn] = C

                # Find all the neighbors of Pn
                PnNeighborPts = self.eps_neighbours(X, Pn, eps)

                # If Pn has at least MinPts neighbors, it's a branch point!
                # Add all of its neighbors to the FIFO queue to be searched.
                if len(PnNeighborPts) >= MinPts:
                    NeighborPts = NeighborPts + PnNeighborPts
            # Advance to the next point in the FIFO queue.
            i += 1

    def plot(self, pl):
        X_noise = self.X[self.C == -1]
        X_clustered = self.X[self.C != -1]
        C_clustered = self.C[self.C != -1]
        vmax = np.unique(C_clustered).shape[0] + 1
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i, data in enumerate(X_clustered):
            circle1 = pl.Circle(data, self.eps, color=m.to_rgba(C_clustered[i]))
            pl.gcf().gca().add_artist(circle1)
        pl.scatter(X_clustered[:, 0], X_clustered[:, 1], c=C_clustered, s=70, zorder=2)
        pl.scatter(X_noise[:, 0], X_noise[:, 1], marker='x', c='w', s=150, edgecolors='black', zorder=2)

    def fit(self, X, eps, MinPts):

        self.X = X
        self.eps = eps
        labels = [0] * len(X)
        # C is the ID of the current cluster.
        C = 0

        # This outer loop is just responsible for picking new seed points--a point
        for P in range(0, len(X)):
            # If the point's label is not 0, continue to the next point.
            if not (labels[P] == 0):
                continue

            # Find all of P's neighboring points.
            NeighborPts = self.eps_neighbours(X, P, eps)
            if len(NeighborPts) < MinPts:
                labels[P] = -1
            # Otherwise, if there are at least MinPts nearby, use this point as the
            # seed for a new cluster.
            else:
                C += 1
                self.extend_cluster(X, labels, P, NeighborPts, C, eps, MinPts)

        # All data has been clustered!
        self.C = np.asarray(labels)
