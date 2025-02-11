#!python3

'''
source: https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py
'''

import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing as preprocessing
import sys
from tqdm import tqdm


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=50, alphas_init=[1, 2], betas_init=[2, 1], weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x_i, loss_max, loss_min):
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def stats(self):
        nu = self.alphas + self.betas
        mu = self.alphas / nu
        var = self.alphas * self.betas / (nu * nu * (nu + 1))
        return mu, var

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


class Beta(object):
    def __init__(self):
        self.alpha, self.beta = -1, -1
        self.xmin, self.xmax = np.nan, np.nan

    def fit(self, x):
        self.xmin, self.xmax = x.min(), x.max()
        x_norm = (x - self.xmin) / (self.xmax - self.xmin)
        mu, var = x_norm.mean(), x_norm.var()
        assert 0 < var < mu * (1 - mu), 'var=%s, mu=%s' % (var, mu)
        self.alpha = mu * (mu * (1 - mu) / var - 1)
        self.beta = (1 - mu) * (mu * (1 - mu) / var - 1)
        return self

    def pdf(self, x):
        x_norm = (x - self.xmin) / (self.xmax - self.xmin)
        # assert (x_norm < 0).sum() <= 0, '%s < %s' % (x.min(), self.xmin)
        # assert (x_norm > 1).sum() <= 0, '%s > %s' % (x.max(), self.xmax)
        x_norm[x_norm < 0] = 0
        x_norm[x_norm > 1] = 1
        return stats.beta.pdf(x_norm, self.alpha, self.beta)

    def stats(self):
        nu = self.alpha + self.beta
        mu = self.alpha / nu
        var = self.alpha * self.beta / (nu * nu * (nu + 1))
        return mu, var

    def __str__(self):
        return 'Beta(a={}, b={}, normalize[{}, {}])'.format(self.alpha, self.beta, self.xmin, self.xmax)


if __name__ == '__main__':
    bmm = BetaMixture1D()
    print(bmm)
    beta = Beta()
    print(beta)
