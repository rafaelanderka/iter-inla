import numpy as np

def mse(x1, x2):
    """
    Compute the mean squared error
    """
    return np.mean((x1 - x2) ** 2)
