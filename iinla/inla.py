import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sksparse.cholmod import cholesky

from iinla.distributions import MarginalGaussianMixture

def sample_parameter_posterior(logpdf, x0, opt_method="Nelder-Mead", sampling_threshold=5,
                               sampling_step_size=2, sampling_evec_scales=None, param_bounds=None, tol=1e-7):
    """
    Sample quadrature nodes in parameter space based on the posterior marginal landscape log(p(θ|y)).

    Return
    ------
    samples: list
        samples_x: parameter samples obtained around the mode of p(θ|y).
        samples_p: normalised value of log(p(θ|y)) at parameter samples.
        samples_mean: posterior mean values μ_{θ|y} at parameter samples.
        samples_vars: posterior marginal variance values σ^2_{θ|y} at parameter samples.
        samples_shift: posterior shift values b_{θ|y} at parameter samples. The shift parameter is defined as b = Σ^{-1}μ.
        samples_precision: precision values P_{θ|y} at parameter samples.
    H_v: array
        eigenvectors of the Hessian of log(p(θ|y)) around the mode.
    x_map:
        the mode of log(p(θ|y)).
    marginal_dist:
        mixture of Gaussians approximating the marginal posteriors p(u_i | y) = Σ_k p(θ_k|y) Δ_k p(u_i | θ_k, y).

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
    print("MAP parameters:", x_map)

    # Calculate eigenvectors of Hessian at mode, to be used as sampling directions
    H = _hessian(neg_logpdf, x_map) # H is (M, M)
    H_w, H_v = eigh(H) # Note H_v is (M, N)

    # The first sample is directly at the mode
    def evaluate_logpdf_full(x):
        # For backward compatibility
        try:
            p0, post_mean, post_vars, post_shift, post_precision = logpdf_full(x)
        except ValueError:
            p0, post_mean, post_vars = logpdf_full(x)
            post_shift = None
            post_precision = None
        return p0, post_mean, post_vars, post_shift, post_precision

    p0, post_mean, post_vars, post_shift, post_precision = evaluate_logpdf_full(x_map)

    samples_x = [x_map]
    samples_p = [p0]
    samples_mean = [post_mean]
    samples_vars = [post_vars]
    samples_shift = [post_shift]
    samples_precision = [post_precision]

    # Sample along eigenvectors of modal Hessian starting from mode
    ranges = [] # Keeps track of the min. and max. steps taken in each direction
    for i, evec in enumerate(H_v.T):
        r = []
        for dir in [-1, 1]:
            offset = dir
            xS = x_map + offset * sampling_step_size * sampling_evec_scales[i] * evec
            if not (xS <= 0).any():
                while p0 - logpdf(xS) < sampling_threshold:
                    break
                    print(offset)
                    pS, post_mean, post_vars, post_shift, post_precision = evaluate_logpdf_full(xS)
                    samples_x.append(xS.copy())
                    samples_p.append(pS)
                    samples_mean.append(post_mean)
                    samples_vars.append(post_vars)
                    samples_shift.append(post_shift)
                    samples_precision.append(post_precision)
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
        pS, post_mean, post_vars, post_shift, post_precision = evaluate_logpdf_full(xS)
        if p0 - pS < sampling_threshold:
            samples_x.append(xS)
            samples_p.append(pS)
            samples_mean.append(post_mean)
            samples_vars.append(post_vars)
            samples_shift.append(post_shift)
            samples_precision.append(post_precision)
    samples_x = np.array(samples_x)
    samples_p = np.array(samples_p)
    samples_mean = np.array(samples_mean)
    samples_vars = np.array(samples_vars)
    samples_shift = np.array(samples_shift)
    samples_precision = np.array(samples_precision)

    # Exponentiate and normalise samples of marginal posterior
    samples_p = np.exp(samples_p - samples_p.mean())
    samples_p /= np.sum(samples_p)

    marginal_dist = MarginalGaussianMixture(samples_p, samples_mean, np.sqrt(samples_vars)) # Only works in moment parameterisation for now

    return [samples_x, samples_p, samples_mean, samples_vars, samples_shift, samples_precision], H_v, x_map, marginal_dist


def compute_field_posterior_stats(samples, parameterisation='moment', calc_std=True):
    """
    Compute the averaged statistics of the state.

    Notes
    -----
    - If parameterisation = 'moment', we compute the averaged posterior mean.
    - If parameterisation = 'natural', we compute the averaged posterior shift and averaged posterior precision, and compute
      the corresponding mean from these quantities.
    - If calc_std = True, we can output the uncertainty estimates. But this is optional, as it does not affect the algorithm.

    """
    # Unpack samples
    samples_x = samples[0]
    samples_p = samples[1]
    samples_mean = samples[2]
    samples_vars = samples[3]
    samples_shift = samples[4]
    samples_precision = samples[5]

    if parameterisation == 'moment':
        # Compute summary statistics
        posterior_mean_marg = (samples_p[:,None,None] * samples_mean).sum(axis=0)
        posterior_marg_second_moment = (samples_p[:,None,None] * (samples_vars + samples_mean**2)).sum(axis=0)
        posterior_var_marg = posterior_marg_second_moment - posterior_mean_marg ** 2
        posterior_std_marg = np.sqrt(posterior_var_marg)
        if calc_std:
            return posterior_mean_marg, posterior_std_marg
        else:
            return posterior_mean_marg
    elif parameterisation == 'natural':
        posterior_shift_marg = (samples_p[:,None] * samples_shift).sum(axis=0)
        posterior_precision_marg = (samples_p * samples_precision).sum()
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
    # Compute the Cartesian product of the lists
    return itertools.product(*range_generators)
