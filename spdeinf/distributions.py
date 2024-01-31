#%%
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class Distribution(ABC):
    """
    Abstract class for distributions.
    """

    @abstractmethod
    def logpdf(self, x):
        return NotImplementedError


class Normal(Distribution):
    """
    Gaussian distribution.
    """
    def __init__(self, mu, sigma):
        self.mu = mu # mean
        self.sigma = sigma # standard deviation

    def logpdf(self, x):
        return stats.norm.logpdf(x, self.mu, self.sigma)


class LogNormal(Distribution):
    """
    Lognormal distribution.
    This is given by Y = e^X.
    """
    def __init__(self, mu, sigma):
        # Note: mu and sigma are the mean and std of the r.v. X
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        return stats.lognorm.logpdf(x, self.sigma, scale=np.exp(self.mu))


class GaussianMixture(Distribution):
    """
    Gaussian mixture distribution.
    The pdf is given by p(x) = \sum_k w_k N(x|μ_k, σ_k).
    """
    def __init__(self, weights, means, stds):
        assert len(weights) == len(means), 'weights must have the same length as means'
        assert len(weights) == len(stds), 'weights must have the same length as stds'

        self.weights = weights # weights of length N
        self.means = means # Shape (N,)
        self.stds = stds # Shape (N,)
    
    def pdf(self, x):
        pdf = [w * stats.norm.pdf(x, self.means[i], self.stds[i]) for i, w in enumerate(self.weights)]
        pdf = np.sum(np.array(pdf), axis=0) # Shape (...)
        return pdf

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def sample(self, num_samples):
        mixture_idx = np.random.choice(len(self.weights), size=num_samples, replace=True, p=self.weights) # choose which Gaussian to sample from
        samples = np.fromiter([stats.norm.rvs(self.means[i], self.stds[i]) for i in mixture_idx], dtype=np.float64) # sample from chosen Gaussians
        return samples


class MarginalGaussianMixture(Distribution):
    """
    Marginal Gaussian mixture on each component.
    The pdf is given by p(x_i) = \sum_k w_k N(x_i|μ_i^k, σ_i^k) for i = 1, ..., D.
    """
    def __init__(self, weights, means, stds):
        shape = means.shape[1:]
        assert len(weights) == len(means), 'weights must have the same length as means'
        assert len(weights) == len(stds), 'weights must have the same length as stds'
        assert stds.shape[1:] == shape, "shape of means must agree with shape of stds"

        self.weights = weights # weights of length N
        self.means = means # Shape (N, ...)
        self.stds = stds # Shape (N, ...)
        self.shape = shape
    
    def pdf(self, x):
        N = len(self.weights)
        assert x.shape == self.shape, "shape of x must agree with shape of means"
        pdf = np.diag(self.weights) @ stats.norm.pdf(x.reshape(1,-1), self.means.reshape(N,-1), self.stds.reshape(N,-1)) # Shape (N, -1)
        pdf = pdf.reshape((N, *self.shape)) # Shape (N, ...)
        pdf = np.sum(pdf, axis=0) # Shape (...)
        return pdf

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def sample(self, num_samples):
        mixture_idx = np.random.choice(len(self.weights), size=num_samples, replace=True, p=self.weights) # Shape (S, ...)
        samples = np.array([stats.norm.rvs(self.means[i], self.stds[i]) for i in mixture_idx]) # Shape (S, ...)
        samples = samples.reshape((num_samples, *self.shape))
        return samples


# %%
