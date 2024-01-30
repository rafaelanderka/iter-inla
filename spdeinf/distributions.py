import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class Distribution(ABC):
    """
    Abstract class for distributions
    """
    @abstractmethod
    def logpdf(self, x):
        return NotImplementedError


class Normal(Distribution):
    """
    Gaussian distribution
    """
    def __init__(self, mu, sigma):
        self.mu = mu # mean
        self.sigma = sigma # standard deviation

    def logpdf(self, x):
        return stats.norm.logpdf(x, self.mu, self.sigma)


class LogNormal(Distribution):
    """
    Lognormal distribution
    This is given by Y = e^X
    """
    def __init__(self, mu, sigma):
        # Note: mu and sigma are the mean and std of the r.v. X
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        return stats.lognorm.logpdf(x, self.sigma, scale=np.exp(self.mu))
