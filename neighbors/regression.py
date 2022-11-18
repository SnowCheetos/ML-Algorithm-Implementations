import numpy as np

class KNNRegressor:

    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def distance(self, X):
        return np.linalg.norm(self.X - X, axis=1)

    def softmax(self, x):
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum() 

    def predict(self, X):
        outputs = []
        for x in X:
            dists = self.distance(x)
            idx = dists.argsort()[:self.k]
            w = self.softmax(dists[idx] / dists[idx].max())
            ys = self.y[idx]
            outputs.append(w @ ys)
        return outputs
