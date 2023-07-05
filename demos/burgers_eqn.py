import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity
from scipy.integrate import odeint
from findiff import FinDiff, Coef
from sksparse.cholmod import cholesky

from spdeinf import nonlinear, plotting, util


## Generate data from Burger's equation

# Define parameters of Burgers' equation
mu = 1
nu = 0.05 # Kinematic viscosity coefficient

# Create spatial discretisation
L_x = 10                      # Range of the domain according to x [m]
dx = 0.1                      # Infinitesimal distance
N_x = int(L_x / dx)           # Points number of the spatial mesh
X = np.linspace(0, L_x, N_x)  # Spatial array

# Create temporal discretisation
L_t = 8                       # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt)           # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array

# Define wave number discretization
k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

# Define the initial condition    
u0 = np.exp(-(X - 3)**2 / 2)

def burgers_odes(u, t, k, mu, nu):
    """
    Construct system of ODEs for Burgers' equation using method of lines
    Note we use a pseudo-spectral method s.t. we construct the (discrete) spatial derivatives in fourier space.
    PDE ---(FFT)---> ODE system
    """
    # Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k**2 * u_hat

    # Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # ODE resolution
    u_t = -mu * u * u_x + nu * u_xx
    return u_t.real

# Solve system of ODEs
u = odeint(burgers_odes, u0, T, args=(k, mu, nu,), mxstep=5000)

# Sample observations
obs_noise = 1e-4
obs_count = 100
obs_dict = util.sample_observations(u, obs_count, obs_noise)
obs_idxs = np.array(list(obs_dict.keys()), dtype=int)


## Fit GP with non-linear PDE prior from Burgers' equation

def get_diff_op(u, dx, dt, nu):
    """
    Constructs current linearised differential operator L for Lu=0.
    """
    return FinDiff(0, dt, 1) + Coef(u) * FinDiff(1, dx, 1) - Coef(nu) * FinDiff(1, dx, 2)

# Perform iterative optimisation
max_iter = 10
diff_op_gen = lambda u: get_diff_op(u, dx, dt, nu)
model = nonlinear.NonlinearSPDERegressor(u, dx, dt, diff_op_gen)
model.fit(obs_dict, obs_noise, max_iter=max_iter, animated=False)

# Check prior covariance
diff_op_init = diff_op_gen(np.zeros_like(u))
diff_op_final = diff_op_gen(model.posterior_mean)

L_init = util.operator_to_matrix(diff_op_init, u.shape, interior_only=False)
L_final = util.operator_to_matrix(diff_op_final, u.shape, interior_only=False)
LL_init = L_init.T @ L_init
LL_final = L_final.T @ L_final
LL_init_chol = cholesky(LL_init + identity(LL_init.shape[0]))
LL_final_chol = cholesky(LL_final + identity(LL_final.shape[0]))
prior_init_std = np.sqrt(LL_init_chol.spinv().diagonal().reshape(u.shape))
prior_final_std = np.sqrt(LL_final_chol.spinv().diagonal().reshape(u.shape))

fig, ax = plt.subplots(1, 2)
im_init = ax[0].imshow(prior_init_std.T, origin="lower")
im_final = ax[1].imshow(prior_final_std.T, origin="lower")
fig.colorbar(im_init)
fig.colorbar(im_final)
plt.show()

# Plot results
plotting.plot_gp_2d(u.T, model.posterior_mean.T, model.posterior_std.T, util.swap_cols(obs_idxs), 'figures/burgers_eqn/burgers_eqn_20_iter.png',
                    mean_vmin=0, mean_vmax=1, std_vmin=0, std_vmax=20,
                    diff_vmin=-0.2, diff_vmax=0.2)
print("Saving animation...")
plotting.animate_images([img.T for img in model.v_hist], util.swap_cols(obs_idxs), "figures/burgers_eqn/burgers_eqn_iter_animation.gif")
