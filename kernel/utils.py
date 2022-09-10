import numpy as np

def linear_kernel(X):
    return X @ X.T

def poly_kernel(X, degree, bias):
    return (X @ X.T + bias) ** degree

def rbf_kernel(X, sigma):
    dists = np.sum((X[:,None]-X[None,:])**2,axis=-1)
    return np.exp(-dists/(2*sigma**2))

def linear_pred(X, X1):
    return X @ X1.T

def poly_pred(X, X1, degree, bias):
    return (X @ X1.T + bias) ** degree

def rbf_pred(X, X1, sigma):
    dists = np.sum((X[:,None]-X1)**2,axis=-1)
    return np.exp(-dists/(2*sigma**2))