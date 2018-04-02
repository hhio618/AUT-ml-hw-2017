import numpy as np

# k means alg
import os

# using euclidean distance
import sys

from base.kmeans import KMeans


def predict_cluster_no(X, centroids):
    C = [np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids.values()]) for x_i in X]
    return np.array(map(lambda x: centroids.keys()[x], C))


class Snapshot:
    def __init__(self, C, measure, measure_name, t):
        self.C = C
        self.measure = measure
        self.measure_name = measure_name
        self.t = t

    def plot(self, X, pl, just_save=False):
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .01  # point in the mesh [x_min, m_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - .05, X[:, 0].max() + .05
        y_min, y_max = X[:, 1].min() - .05, X[:, 1].max() + .05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        centroids = {}
        for cluster_no in np.unique(self.C):
            idx = (self.C == cluster_no)
            centroids[cluster_no] = X[idx].mean(axis=0)
        Z = predict_cluster_no(np.c_[xx.ravel(), yy.ravel()], centroids)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure(1)
        pl.clf()
        pl.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=pl.cm.Paired,
                  aspect='auto', origin='lower')

        pl.plot(X[:, 0], X[:, 1], 'k.', markersize=10)
        # Plot the centroids as a white X
        centroids = np.array(centroids.values())
        pl.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)
        pl.title('Top down K-means clustering')
        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())
        if just_save:
            pl.savefig("outputs%smain2%s%s%ssnapshot@%d.png" % (os.sep, os.sep, self.measure_name, os.sep, self.t))
            # pl.scatter(self.X[:, 0], self.X[:, 1], c=self.C, s=50)
            # pl.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black',marker='x',s=100)

    def save_fig(self, X, pl):
        self.plot(X, pl, just_save=True)


class TDKMeans(object):
    class SSE:
        def __repr__(self):
            return "sse"

        def __call__(self, *args, **kwargs):
            X = args[0]
            C = args[1]
            X_copy = X.copy()
            for cluster_no in np.unique(C):
                idx = (C == cluster_no)
                X_copy[idx] = X_copy[idx] - X_copy[idx].mean(axis=0)
            return np.linalg.norm(X_copy)

    def __init__(self, level, iter_count, measure):
        self.level = level
        self.iter_count = iter_count
        self.X = None
        self.C = None
        self.measure = measure

    # train kmeans and find clusters and centeroids
    def fit(self, X):
        self.X = X
        self.C = 0 * np.ones(X.shape[0], dtype=int)
        snapshots = [Snapshot(self.C, self.measure(self.X, self.C), str(self.measure), 1)]
        print "Level %d, best measure(%s) --> %f" % (1, str(self.measure), snapshots[0].measure)
        for t in range(2, self.level + 1):
            test_list = []
            for cluster_no in np.unique(self.C):
                C_copy = self.C.copy()
                idx_test = (C_copy == cluster_no)
                model = KMeans(2, self.iter_count).fit(self.X[idx_test])
                C_copy[idx_test] = model.C + self.C.max() + 1  # <----- cluster number shoud be different
                test_list.append((C_copy, self.measure(self.X, C_copy), str(self.measure), t))

            snap = min(test_list, key=lambda item: item[1])
            print "Level %d, best measure(%s) --> %f" % (t, str(self.measure), snap[1])
            snapshots.append(Snapshot(*snap))
            self.C = snap[0]
        return snapshots
