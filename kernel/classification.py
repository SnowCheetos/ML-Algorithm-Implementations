import numpy as np
from utils import *

class KernelBinaryClassifier:

    def __init__(self, kernel='linear'):
        self.kernels = {
            'linear': (linear_kernel, linear_pred),
            'poly': (poly_kernel, poly_pred),
            'rbf': (rbf_kernel, rbf_pred)
        }
        self.kernel = kernel
        self.X = None
        self.y = None
        self.K = None

    def fit(self, X, y, **kwargs):
        self.X = X
        self.y = y
        if self.kernel == 'linear':
            self.K = self.kernels[self.kernel][0](X)
        elif self.kernel == 'poly':
            self.K = self.kernels[self.kernel][0](X, kwargs['degree'], kwargs['bias'])
        elif self.kernel == 'rbf':
            self.K = self.kernels[self.kernel][0](X, kwargs['sigma'])
    
    def predict(self, X, Lambda=1, **kwargs):
        if self.kernel == 'linear':
            K = self.kernels[self.kernel][1](self.X, X)
        elif self.kernel == 'poly':
            K = self.kernels[self.kernel][1](self.X, X, kwargs['degree'], kwargs['bias'])
        elif self.kernel == 'rbf':
            K = self.kernels[self.kernel][1](self.X, X, kwargs['sigma'])
        pred = self.y.T @ np.linalg.inv(self.K + Lambda*np.eye(K.shape[0])) @ K
        return (pred > .5).astype(int)