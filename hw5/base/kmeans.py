import numpy as np

# k means alg
import sys


class KMeans(object):
    def __init__(self, K, iter_count):
        self.K = K
        self.iter_count = iter_count
        self.centroids = None
        self.X = None
        self.C = None

    # train kmeans and find clusters and centeroids
    def fit(self, X):
        self.X = X
        idx = np.arange(self.X.shape[0])
        np.random.shuffle(idx)
        self.centroids = self.X[idx[:self.K]]
        for i in xrange(0, self.iter_count):
            # Cluster Assignment step
            self.C = self._euclidean(self.X)
            # Move centroids step
            self.centroids = [self.X[self.C == k].mean(axis=0) for k in range(self.K)]
        self.centroids = np.array(self.centroids)
        return self

    def _s(self, i):
        X_i = self.X[self.C == i]
        S = 0.
        for x in X_i:
            S += np.linalg.norm(x - self.centroids[i]) ** 2
        return S / X_i.shape[0] ** 1. / 2.

    def _m(self, i, j):
        return np.linalg.norm(self.centroids[i] - self.centroids[j])

    def _r(self, i, j):
        return float(self._s(i) + self._s(j)) / self._m(i, j)

    def dbi(self):
        dbi = 0
        C = range(self.K)
        for i in C:
            D_i = -sys.maxsize
            C_prime = range(self.K)
            C_prime.remove(i)
            for j in C_prime:
                D_i = max(D_i, self._r(i, j))
            dbi += D_i
        return dbi / float(self.K)

    # compute euclidean distance
    def _euclidean(self, X):
        return np.array([np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in self.centroids]) for x_i in X])

    # predict each cluster label according to their datas
    def predict(self, X):
        return self._euclidean(X)

    def plot(self, pl):
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .01  # point in the mesh [x_min, m_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = self.X[:, 0].min() - .3, self.X[:, 0].max() + .3
        y_min, y_max = self.X[:, 1].min() - .3, self.X[:, 1].max() + .3
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure(1)
        pl.clf()
        pl.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=pl.cm.Paired,
                  aspect='auto', origin='lower')

        pl.plot(self.X[:, 0], self.X[:, 1], 'k.', markersize=10)
        # Plot the centroids as a white X
        centroids = self.centroids
        pl.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)
        pl.title('K-means clustering results')
        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())

        # pl.scatter(self.X[:, 0], self.X[:, 1], c=self.C, s=50)
        # pl.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black',marker='x',s=100)
