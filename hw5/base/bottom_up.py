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
    def __init__(self, C, num_clusters, measure_name, linkage_measure, cluster_i, cluster_j):
        self.C = C
        self.num_clusters = num_clusters
        self.linkage_measure = linkage_measure
        self.measure_name = measure_name
        self.merges = (cluster_i, cluster_j)

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
        pl.title('Top Bottom K-means clustering')
        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())
        if just_save:
            pl.savefig(
                "outputs%smain3%s%s%ssnapshot@%d.png" % (os.sep, os.sep, self.measure_name, os.sep, self.num_clusters))
            # pl.scatter(self.X[:, 0], self.X[:, 1], c=self.C, s=50)
            # pl.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black',marker='x',s=100)

    def save_fig(self, X, pl):
        self.plot(X, pl, just_save=True)


class DistanceMatrix(dict):
    def __init__(self, **kwargs):
        super(DistanceMatrix, self).__init__(**kwargs)

    def delete(self, row, cluster_ids):
        for id in cluster_ids:
            self.pop((min(row, id), max(row, id)))


class AgglomerativeClustering(object):
    def __init__(self, level, linkage_measure):
        self.level = level
        self.X = None
        self.C = None
        self.linkage_measure = linkage_measure

    # train kmeans and find clusters and centeroids
    def fit(self, X):
        self.X = X
        self.C = np.arange(0, self.X.shape[0], dtype=int)
        snapshots = []
        num_clusters = self.X.shape[0]
        cluster_ids = np.unique(self.C)
        # initial distances
        dm = DistanceMatrix()

        for cluster_i in cluster_ids:
            for cluster_j in cluster_ids:
                if cluster_i != cluster_j:
                    X_i = self.X[self.C == cluster_i]
                    X_j = self.X[self.C == cluster_j]
                    dm[(min(cluster_i, cluster_j), max(cluster_i, cluster_j))] = self.linkage_measure(X_i, X_j)

        while num_clusters != 1:
            best_merge = min(dm.keys(), key=(lambda k: dm[k]))
            self.merge(*best_merge)
            cluster_ids = np.unique(self.C)
            measure = dm[best_merge]
            dm.delete(best_merge[1], cluster_ids)
            X_merge = self.X[self.C == best_merge[0]]
            for cluster_i in cluster_ids:
                X_i = self.X[self.C == cluster_i]
                if cluster_i < best_merge[0]:
                    dm[(cluster_i, best_merge[0])] = self.linkage_measure(X_merge, X_i)
                elif cluster_i > best_merge[0]:
                    dm[(best_merge[0], cluster_i)] = self.linkage_measure(X_merge, X_i)
            print "#Clusters: %d, best linkage(%s) --> %f" % (num_clusters, str(self.linkage_measure), measure)
            num_clusters -= 1
            if num_clusters <= self.level:
                snapshots.append(Snapshot(self.C.copy(), num_clusters, str(self.linkage_measure), measure,
                                          *best_merge))
        return snapshots

    def merge(self, cluster_i, cluster_j):
        cluster_min_idx = min(cluster_i, cluster_j)
        cluster_max_idx = max(cluster_i, cluster_j)
        self.C[self.C == cluster_max_idx] = cluster_min_idx
