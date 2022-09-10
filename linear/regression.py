import numpy as np

class LinearRegressor:

    def __init__(self, bias=True, regularization=True):
        self.bias = bias
        self.reg = regularization
        self.w = None

    def fit(self, X, y, Lambda=1):
        if self.bias:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)
        if self.reg:
            self.w = np.linalg.inv(X.T @ X + Lambda * np.eye(X.shape[1])) @ X.T @ y
        else:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        if self.bias:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)
        return X @ self.w


class PolynomialRegressor:
    """
    Not recommended for degrees higher than 2, use kernel methods instead.
    """
    def __init__(self, degree, bias=True, regularization=True):
        self.bias = bias
        self.degree = degree
        self.reg = regularization
        self.w = None

    def square_expansion(self, X):
        return np.einsum('ij,ik->ijk', X, X)[:, np.triu_indices(X.shape[1])[0], np.triu_indices(X.shape[1])[1]]

    def feature_expansion(self, X):
        if self.degree == 1:
            return X
        elif self.degree == 2:
            return np.concatenate([X, self.square_expansion(X)], 1)
        else:
            X1 = X
            X2 = self.square_expansion(X)
            XP = X2
            X = np.concatenate([X1, X2], 1)
            for d in range(self.degree - 2):
                    XP = np.einsum('ij,ik->ijk', X1, XP)
                    XP = XP.reshape(-1, XP.shape[1] * XP.shape[2])
                    X = np.concatenate([X, np.tril(XP)], 1)
            return X

    def fit(self, X, y, Lambda=1):
        X = self.feature_expansion(X)
        if self.bias:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)
        if self.reg:
            self.w = np.linalg.inv(X.T @ X + Lambda * np.eye(X.shape[1])) @ X.T @ y
        else:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y


    def predict(self, X):
        X = self.feature_expansion(X)
        if self.bias:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], 1)
        return X @ self.w