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
        max_x = np.amax(x, 1).reshape(x.shape[0],1)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1, keepdims=True) 

    def predict(self, X):
        outputs = []
        for x in X:
            dists = self.distance(x)
            idx = dists.argsort()[:self.k]
            w = dists[idx] / dists[idx].max()
            ys = self.y[idx]
            outputs.append(w @ ys)
        return outputs
