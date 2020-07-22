import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)

        # init mean, var, priors




    def predict(self, X):
        pass