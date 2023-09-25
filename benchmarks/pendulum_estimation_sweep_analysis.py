import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from spdeinf import metrics

def moving_median(signal, window_size):
    medians = []
    for i in range(len(signal) - window_size + 1):
        medians.append(np.median(signal[i:i+window_size]))
    return np.array(medians)

def moving_average(signal, window_size):
    padding = window_size // 2
    data_padded = np.pad(signal, (padding, padding), 'edge')
    return np.convolve(data_padded, np.ones(window_size)/window_size, mode='valid')

def moving_std(signal, window_size):
    padding = window_size // 2
    data_padded = np.pad(signal, (padding, padding), 'edge')
    return [np.std(data_padded[i:i+window_size]) for i in range(len(signal))]

with open("data/pendulum_estimation_sweep_opt_coarse.pkl", 'rb') as f:
    data = pickle.load(f)
t = 0.1 * data['windows']

for data_file, desc in [("data/pendulum_estimation_sweep_opt_coarse.pkl", "coarse"), ("data/pendulum_estimation_sweep_opt_fine.pkl", "fine")]:
    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    b = data['b_means']
    c = data['c_means']

    # Find and replace outliers with median
    b_threshold = 0.05
    b_median = moving_median(b, 5)
    b_outliers = np.abs(b[len(b)-len(b_median):] - b_median) > b_threshold
    print(b_outliers.any())
    b_cleaned = np.copy(b)
    b_cleaned[len(b)-len(b_median):][b_outliers] = b_median[b_outliers]

    c_threshold = 0.04
    c_median = moving_median(c, 5)
    c_outliers = np.abs(c[len(c)-len(c_median):] - c_median) > c_threshold
    c_cleaned = np.copy(c)
    c_cleaned[len(c)-len(c_median):][c_outliers] = c_median[c_outliers]

    # Calculate the moving average and standard deviation
    window_size = 5
    b_avg = moving_average(b_cleaned, window_size)
    b_std = moving_std(b_cleaned, window_size)
    c_avg = moving_average(c_cleaned, window_size)
    c_std = moving_std(c_cleaned, window_size)

    # Plotting
    plt.figure(figsize=(6,3))
    avg_handle = plt.plot(t, b_avg, color='k', label=f"{window_size}-step moving average")[0]
    std_handle = plt.fill_between(t, b_avg - b_std, b_avg + b_std, color='grey', alpha=0.5, label=f"{window_size}-step std. dev.")
    true_handle = plt.axhline(0.2, linestyle=':', c='r', label='True value')
    plt.xlabel("Training window length")
    plt.ylabel("$b$", labelpad=5)
    plt.ylim((-0.0789, 0.330))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    print(plt.ylim())
    plt.tight_layout(pad=0.5)
    plt.savefig(f"figures/pendulum/pendulum_estimation_sweep_b_{desc}.pdf", transparent=True)

    plt.figure(figsize=(6,3))
    plt.plot(t, c_avg, color='k', label=f"{window_size}-step moving average")
    plt.fill_between(t, c_avg - c_std, c_avg + c_std, color='grey', alpha=0.5, label=f"{window_size}-step std. dev.")
    plt.axhline(1.0, linestyle=':', c='r', label='True value')
    plt.xlabel("Training window length")
    plt.ylabel("$c$", labelpad=5)
    print(plt.ylim())
    plt.ylim((0.167, 1.53))
    plt.tight_layout(pad=0.5)
    plt.savefig(f"figures/pendulum/pendulum_estimation_sweep_c_{desc}.pdf", transparent=True)

# Define damped pendulum eqn. and observation parameters
b = 0.2
c = 1.

# Create temporal discretisation
L_t = 24.3                    # Duration of simulation [s]
dt = 0.1                      # Infinitesimal time
N_t = int(L_t / dt) + 1       # Points number of the temporal mesh
T = np.linspace(0, L_t, N_t)  # Temporal array
T = np.around(T, decimals=1)

# Define the initial condition    
u0 = [0.75 * np.pi, 0.]

# Define corresponding system of ODEs
def pend(u, t, b, c):
    theta, omega = u
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt

u_full = odeint(pend, u0, T, args=(b, c,))
u = u_full[:, 0]

# Visualise testing procedure
training_cutoff = 90
plt.figure(figsize=(6, 3))
gt_handle = plt.plot(T, u, c='b', zorder=3, label="Ground truth")[0]
cutoff_handle = plt.axvline(T[training_cutoff], c='grey', linestyle=':', label="Training window cutoff")
obs_handle = plt.scatter(T[:training_cutoff], u[:training_cutoff], c='r', marker='x', label="Observations")
plt.xlabel("$t$")
plt.ylabel("$u$", labelpad=5)
plt.annotate(xy=(T[training_cutoff] - 2,1.8), xytext=(T[training_cutoff] + 2,1.8), arrowprops=dict(arrowstyle='<->'), text="")
plt.text(T[training_cutoff] - 6.6, 2.2, 'Training window') 
plt.tight_layout(pad=0.5)
plt.savefig("figures/pendulum/pendulum_estimation_sweep.pdf", transparent=True)

# Generate legends
fig, ax = plt.subplots(2, 2, figsize=(6,3), width_ratios=[0.5, 0.5], height_ratios=[0.35, 0.65])
ax = ax.flatten()
for i in range(len(ax)):
    if i == 0 or i == 2: continue
    ax[i].remove()
ax[0].legend(loc="center", handles=[gt_handle, cutoff_handle, obs_handle])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].get_xaxis().set_ticks([])
ax[0].get_yaxis().set_ticks([])

handle_fill = matplotlib.collections.PolyCollection([])
handle_fill.update_from(std_handle)
ax[2].legend(loc="center", handles=[avg_handle, handle_fill, true_handle])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)
ax[2].spines['left'].set_visible(False)
ax[2].get_xaxis().set_ticks([])
ax[2].get_yaxis().set_ticks([])
plt.savefig("figures/pendulum/pendulum_estimation_sweep_legends.pdf", transparent=True)
plt.show()
