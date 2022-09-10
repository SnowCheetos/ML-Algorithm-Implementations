import numpy as np

class MSELoss:

    def __init__(self):
        self.dx = None

    def forward(self, pred, y):
        return (1/y.shape[0]) * np.linalg.norm(pred - y)**2

    def backward(self, pred, y):
        self.dx = 2*(1/y.shape[0])*(pred - y)


class CELoss:

    def __init__(self):
        pass