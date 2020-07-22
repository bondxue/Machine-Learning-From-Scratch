import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]

        # extract the labels of the k nearest neighbor training samples
        k_nearest_neighbor_labels = [self.y_train[i] for i in k_idx]

        # return the most common class label
        most_common = Counter(k_nearest_neighbor_labels).most_common(1)
        return most_common[0][0]

    @staticmethod
    def _euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

