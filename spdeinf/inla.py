import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from scipy import sparse
from sksparse.cholmod import cholesky

from spdeinf import linear, nonlinear, util


def sample_parameter_posterior(logpdf, x0, opt_method="Nelder-Mead", sampling_threshold=5,
                               sampling_step_size=2, sampling_evec_scales=None, param_bounds=None, tol=1e-7):
    """
    param_1 = mean or shift
    param_2 = vars or precision
    """
    # Process args
    if not sampling_evec_scales:
        sampling_evec_scales = len(x0) * [1]
    sampling_evec_scales = np.array(sampling_evec_scales)

    # Make some convenient aliases for the log PDF of the parameter posterior
    neg_logpdf = lambda x: -logpdf(x)
    logpdf_full = lambda x: logpdf(x, return_conditional_params=True)

    # Find the mode of the parameter posterior
    print("Finding parameter posterior mode...")
    opt = minimize(fun=neg_logpdf, x0=x0, method=opt_method, bounds=param_bounds)
    x_map = opt["x"]
    print(opt)

    # Calculate eigenvectors of Hessian at mode, to be used as sampling directions
    H = _hessian(neg_logpdf, x_map) # H is (M, M)
    H_w, H_v = eigh(H) # Note H_v is (M, N)

    # The first sample is directly at the mode
    p0, post_param_1, post_param_2 = logpdf_full(x_map)
    samples_x = [x_map]
    samples_p = [p0]
    samples_param_1 = [post_param_1]
    samples_param_2 = [post_param_2]

    # Sample along eigenvectors of modal Hessian starting from mode
    ranges = [] # Keeps track of the min. and max. steps taken in each direction
    for i, evec in enumerate(H_v.T):
        r = []
        for dir in [-1, 1]:
            offset = dir
            xS = x_map + offset * sampling_step_size * sampling_evec_scales[i] * evec
            if not (xS <= 0).any():
                while p0 - logpdf(xS) < sampling_threshold:
                    # break
                    pS, post_param_1, post_param_2 = logpdf_full(xS)
                    samples_x.append(xS.copy())
                    samples_p.append(pS)
                    samples_param_1.append(post_param_1)
                    samples_param_2.append(post_param_2)
                    offset += dir
                    xS = x_map + offset * sampling_step_size * sampling_evec_scales[i] * evec
                    if (xS <= 0).any():
                        continue
                    # print(sampling_step_size * dir * evec)
            r.append(offset - dir)
        ranges.append(r)
    print("Sampling offset ranges:", ranges)

    # Sample at combinations of eigv. offsets within our min. and max. ranges
    for offset in _generate_combinations(ranges):
        if 0 in offset:
            continue
        xS = x_map + sampling_step_size * (sampling_evec_scales * offset * H_v).sum(axis=1)
        if (xS <= 0).any():
            continue
        pS, post_param_1, post_param_2 = logpdf_full(xS)
        if p0 - pS < sampling_threshold:
            samples_x.append(xS)
            samples_p.append(pS)
            samples_param_1.append(post_param_1)
            samples_param_2.append(post_param_2)
    samples_x = np.array(samples_x)
    samples_p = np.array(samples_p)
    samples_param_1 = np.array(samples_param_1)
    samples_param_2 = np.array(samples_param_2)

    # Exponentiate and normalise samples of marginal posterior
    samples_p = np.exp(samples_p - samples_p.mean())
    samples_p /= np.sum(samples_p)
    return [samples_x, samples_p, samples_param_1, samples_param_2], H_v, x_map


def compute_field_posterior_stats(samples, parameterisation='natural', calc_std=False):
    """
    param_1 = mean or shift
    param_2 = vars or precision
    """
    # Unpack samples
    samples_x = samples[0]
    samples_p = samples[1]
    samples_param_1 = samples[2]
    samples_param_2 = samples[3]

    if parameterisation == 'moment':
        # Compute summary statistics
        posterior_mean_marg = (samples_p[:,None,None] * samples_param_1).sum(axis=0)
        posterior_marg_second_moment = (samples_p[:,None,None] * (samples_param_2**2 + samples_param_1**2)).sum(axis=0)
        posterior_var_marg = posterior_marg_second_moment - posterior_mean_marg ** 2
        posterior_std_marg = np.sqrt(posterior_var_marg)
        if calc_std:
            return posterior_mean_marg, posterior_std_marg
        else:
            return posterior_mean_marg
    elif parameterisation == 'natural':
        posterior_shift_marg = (samples_p[:,None] * samples_param_1).sum(axis=0)
        posterior_precision_marg = (samples_p * samples_param_2).sum()
        posterior_precision_cholesky = cholesky(posterior_precision_marg)
        posterior_mean_marg = posterior_precision_cholesky(posterior_shift_marg)
        if calc_std:
            posterior_var_marg = posterior_precision_cholesky.spinv().diagonal()
            posterior_std_marg = np.sqrt(posterior_var_marg)
            return posterior_mean_marg, posterior_std_marg
        else:
            return posterior_mean_marg


def _hessian(fun, x, epsilon=1e-5):
    """
    Calculate hessian using finite differences
    """
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        x1 = np.array(x, copy=True)
        x2 = np.array(x, copy=True)
        x1[i] += epsilon
        x2[i] -= epsilon
        hess[i, i] = (fun(x1) - 2 * fun(x) + fun(x2)) / (epsilon ** 2)
        for j in range(i+1, n):
            x1[i] -= epsilon
            x1[j] += epsilon
            x2[i] += epsilon
            x2[j] -= epsilon
            hess[i, j] = (fun(x1) - fun(x2)) / (4 * epsilon ** 2)
            hess[j, i] = hess[i, j]
    return hess


def _generate_combinations(ranges):
    # Convert the (min, max)-tuples to range generators
    range_generators = [range(start, end+1) for start, end in ranges]
    # Compute the cartesian product of the lists
    return itertools.product(*range_generators)
