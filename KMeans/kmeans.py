import numpy as np

np.random.seed(42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closet centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)


    def _creat_clusters(self, centroids):
        # Assign the samples to the closest centroids to  create clusters
        clusters = [[] for _ in range(self.K)]

    def _closest_centroid(self, sample, centroids):
        pass

    def _get_cluster_labels(self, clusters):
        pass

    def _create_clusters(self, centroids):
        pass
