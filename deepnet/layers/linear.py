import numpy as np

class LinearLayer:

    def __init__(self, inp_dim, out_dim, bias=True):
        self.bias = bias
        if bias:
            self.w = np.random.uniform(-1, 1, (inp_dim+1, out_dim))
        else:
            self.w = np.random.uniform(-1, 1, (inp_dim, out_dim))

        self.dw = None
        self.dx = None

    def forward(self, x):
        if self.bias:
            x = np.concatenate([np.ones((1, x.shape[0])), x], 1)
            return np.einsum("bi,ij->bj", x, self.w)
        return np.einsum("bi,ij->bj", x, self.w)

    def backward(self, x):
        self.dw = x
        self.dx = self.w

    def update(self, grad, lr=1e-3):
        self.w -= lr * grad