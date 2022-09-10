import numpy as np

class GraphConvLayer:

    def __init__(self, inp_dim, out_dim, bias=True):
        self.bias = bias
        if bias:
            self.w = np.random.uniform(-1, 1, (inp_dim+1, out_dim))
        else:
            self.w = np.random.uniform(-1, 1, (inp_dim, out_dim))
        self.dw = None
        self.da = None
        self.dx = None

    def degree_mat(self, adj):
        return np.diag(adj.sum(1))

    def forward(self, x, adj):
        if self.bias:
            x = np.concatenate([np.ones((1, x.shape[0])), x], 1)
        D_sq = self.degree_mat(adj)**-0.5
        h = D_sq @ adj @ D_sq @ x
        return np.einsum("bi,ij->bj", h, self.w)

    def backward(self, x, adj):
        pass