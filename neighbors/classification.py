import numpy as np

class KNNClassifier:

    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def distance(self, X):
        return np.linalg.norm(self.X - X, axis=1)

    def predict(self, X):
        outputs = []
        for x in X:
            dists = self.distance(x)
            idx = dists.argsort()[:self.k]
            ys = self.y[idx]
            u, c = np.unique(ys, return_counts=True)
            outputs.append(u[c.argmax()])
        return outputs