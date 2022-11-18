import numpy as np

class Tanh:

    def __init__(self):
        self.dx = None

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        self.dx = 1/np.cosh(x)**2


class Sigmoid:

    def __init__(self):
        self.dx = None

    def forward(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, x):
        self.dx = self.forward(x) * (1 - self.forward(x))


class ReLU:

    def __init__(self):
        self.dx = None

    def forward(self, x):
        x[x<0] = 0
        return x

    def backward(self, x):
        self.dx = np.sign(self.forward(x))


class Softmax:

    def __init__(self):
        self.dx = None

    def forward(self, x):
        max_x = np.amax(x, 1).reshape(x.shape[0],1)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1, keepdims=True) 

    def Softmax_grad(self, x):
        s = self.forward(x)
        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]))
        temp1 = np.einsum('ij,jk->ijk', s, a)
        temp2 = np.einsum('ij,ik->ijk', s, s)
        self.dx = temp1 - temp2