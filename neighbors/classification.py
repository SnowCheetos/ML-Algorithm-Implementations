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
        return np.sqrt(np.sum((self.X[:,None]-X)**2,axis=-1))

    def predict(self, X):
        dists = self.distance(X).T
        ranks = np.argsort(dists)
        preds = []
        for r in ranks:
            ys = []
            for k in range(self.k):
                ys.append(self.y[np.where(r == k)[0][0]])
            values, counts = np.unique(np.array(ys), return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)