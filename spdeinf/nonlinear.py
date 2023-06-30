import numpy as np

from . import linear
from . import metrics

def fit_pde_gp(u, diff_op_generator, obs_dict, obs_noise):
    # Guess mean is 0
    v = np.zeros_like(u)

    # Iterative linearisation
    max_iter = 20
    v_hist = [v]
    mixing_coeff = 1
    persist_coeff = 1 - mixing_coeff
    for i in range(max_iter - 1):
        # diff_op_guess = FinDiff(0, dt, 1) + Coef(v) * FinDiff(1, dx, 1) + Coef(diff_op_x(v)) * Identity() - Coef(nu) * FinDiff(1, dx, 2)
        diff_op_guess = diff_op_generator(v)
        res = linear.fit_spde_gp(u, obs_dict, obs_noise, diff_op_guess)
        v = persist_coeff * v + mixing_coeff * res['posterior_mean'].copy()
        v_hist.append(v)
        print(f'iter={i+1:d}, MSE={metrics.mse(u, v)}')

    # Generate final posterior including std. dev.
    diff_op_guess = diff_op_generator(v)
    res = linear.fit_spde_gp(u, obs_dict, obs_noise, diff_op_guess, calc_std=True)
    post_mean = res['posterior_mean']
    post_std = res['posterior_std']
    print(f'iter={max_iter:d}, MSE={metrics.mse(u, post_mean)}')
    v_hist.append(post_mean)
    return post_mean, post_std, v_hist
