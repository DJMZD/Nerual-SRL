import numpy as np
import theano
import theano.tensor as T


def build_shared_zeros(shape):
    return theano.shared(value=np.zeros(shape, dtype=theano.config.floatX),
                         borrow=True)


def get_uniform_weight(x_dim, y_dim=0, low=-0.08, high=0.08):
    if y_dim == 0:
        W = np.random.uniform(low=low, high=high, size=x_dim)
    else:
        W = np.random.uniform(low=low, high=high, size=(x_dim, y_dim))

    return W.astype(theano.config.floatX)


def get_bengio_uniform_weight(x_dim, y_dim=0):
    if y_dim == 0:
        W = np.random.uniform(low=-np.sqrt(6.0 / x_dim),
                              high=np.sqrt(6.0 / x_dim),
                              size=x_dim)
    else:
        W = np.random.uniform(low=-np.sqrt(6.0 / (x_dim + y_dim)),
                              high=np.sqrt(6.0 / (x_dim + y_dim)),
                              size=(x_dim, y_dim))

    return W.astype(theano.config.floatX)


def get_orthogonal_weight(x_dim):
    W = np.random.randn(x_dim, x_dim)
    u, s, v = np.linalg.svd(W)

    return u.astype(theano.config.floatX)


def logsumexp(x, axis=0):
    x_max = T.max(x, axis=axis, keepdims=True)

    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def norm(x, L=2):
    hypotenuse = 0.0
    for p in x:
        hypotenuse += (p ** 2).sum()

    return hypotenuse.sqrt()
