# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import numpy as np

class LR(object):
    @staticmethod
    def fn(w, x):
        '''sigmoid function
        '''
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        '''Cross entropy loss function
        '''
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        '''The first derivative of the cross entropy loss function to the weight W
        '''
        return (y_hat - y) * x


class FTRL(object):
    def __init__(self, dim, l1, l2, alpha, beta, decisionFunc=LR):
        self.dim = dim
        self.decisionFunc = decisionFunc
        self.z = np.zeros(dim)
        self.n = np.zeros(dim)
        self.w = np.zeros(dim)
        self.w_list = []
        self.loss_list = []
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1 else (np.sign(
            self.z[i]) * self.l1 - self.z[i]) / (self.l2 + (self.beta + np.sqrt(self.n[i])) / self.alpha) for i in xrange(self.dim)])
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * self.w
        self.n += g * g
        return self.decisionFunc.loss(y, y_hat)

    def train(self, trainSet, verbos=False, max_itr=10000000000, eta=0.01, epochs=100):
        itr = 0
        n = 0
        while True:
            for x, y in trainSet:
                loss = self.update(x, y)
                if verbos and n%verbos==0:
                    print "itr=" + str(n) + "\tloss=" + str(loss)
                    self.w_list.append(self.w)
                    self.loss_list.append(loss)
                if loss < eta:
                    itr += 1
                else:
                    itr = 0
                if itr >= epochs:  # when the loss function has been continuously epochs iterations less than eta
                    print "loss have less than", eta, " continuously for ", itr, "iterations"
                    return
                n += 1
                if n >= max_itr:
                    print "reach max iteration", max_itr
                    return