from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from nn_utils import build_shared_zeros


class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.updates = {}
        self.grads = []

    def update(self, loss):
        self.updates = OrderedDict()
        self.grads = [T.grad(loss, p) for p in self.parameters]


class SGDOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate=0.01):
        super(SGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate

    def update(self, loss):
        super(SGDOptimizer, self).update(loss)
        for p, g in zip(self.parameters, self.grads):
            self.updates[p] = p - self.learning_rate * g

        return self.updates


class AdamOptimizer(Optimizer):
    def __init__(self, parameters, alpha=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super(AdamOptimizer, self).__init__(parameters)
        # TODO: really?
        self.t = theano.shared(np.float32(1))
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [build_shared_zeros(p.shape.eval()) for p in self.parameters]
        self.v = [build_shared_zeros(p.shape.eval()) for p in self.parameters]

    def update(self, loss):
        super(AdamOptimizer, self).update(loss)
        # TODO: research for this line
        # self.beta1_t = self.beta1 * (1 - 1e-8) ** (self.t - 1)
        for m_tm1, v_tm1, p_tm1, g_tm1 in zip(self.m, self.v,
                                              self.parameters, self.grads):
            m_t = self.beta1 * m_tm1 + (1 - self.beta1) * g_tm1
            v_t = self.beta2 * v_tm1 + (1 - self.beta2) * T.sqr(g_tm1)
            m_hat_t = m_t / (1 - self.beta1 ** self.t)
            v_hat_t = v_t / (1 - self.beta2 ** self.t)
            p_t = p_tm1 - self.alpha * m_hat_t / (T.sqrt(v_hat_t) + self.eps)
            self.updates[p_tm1] = p_t
            self.updates[m_tm1] = m_t
            self.updates[v_tm1] = v_t
        self.updates[self.t] = self.t + 1.0

        return self.updates
