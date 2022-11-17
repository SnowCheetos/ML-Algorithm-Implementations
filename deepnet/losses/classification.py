import numpy as np

class CrossEntropy:

    def __init__(self):
        self.dx = None

    def forward(self, pred, y):
        return -np.sum(y * np.log(pred))

    def backward(self, pred, y):
        grad = pred.copy()
        for i, yb, pd in enumerate(zip(y, pred)):
            for j, b, p in enumerate(zip(yb, pd)):
                if b == p:
                    grad[i, j] -= 1
        self.dx = grad
