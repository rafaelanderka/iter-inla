import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.core.display import HTML

def plot_gp_2d(gt, gp_mean, gp_std, obs_idxs, output_filename, mean_vmin=None, mean_vmax=None,
               std_vmin=None, std_vmax=None, diff_vmin=None, diff_vmax=None):
    fig, axs = plt.subplots(1, 4, figsize=(15,5.3))
    gtim = axs[0].imshow(gt, vmin=mean_vmin, vmax=mean_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[0].set_title('ground truth')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('x')
    fig.colorbar(gtim)
    ptim = axs[1].imshow(gp_mean, vmin=mean_vmin, vmax=mean_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[1].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    axs[1].set_title('mean')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('x')
    fig.colorbar(ptim)
    ptstdim = axs[2].imshow(gp_std, vmin=std_vmin, vmax=std_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[2].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    axs[2].set_title('standard deviation')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('x')
    fig.colorbar(ptstdim)
    diffim = axs[3].imshow(gt - gp_mean, vmin=diff_vmin, vmax=diff_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[3].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    axs[3].set_title('diff')
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('x')
    fig.colorbar(diffim)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

def init_gp_2d_plot(num_runs):
    fig, axs = plt.subplots(num_runs, 4, figsize=(8, num_runs*2.5))
    for i in range(num_runs):
        axs[i, 0].set_title('ground truth')
        axs[i, 0].set_xlabel('time')
        axs[i, 0].set_ylabel('x')
        axs[i, 1].set_title('mean')
        axs[i, 1].set_xlabel('time')
        axs[i, 1].set_ylabel('x')
        axs[i, 2].set_title('standard deviation')
        axs[i, 2].set_xlabel('time')
        axs[i, 2].set_ylabel('x')
        axs[i, 3].set_title('diff')
        axs[i, 3].set_xlabel('time')
        axs[i, 3].set_ylabel('x')
    return fig, axs

def add_gp_2d_plot(fig, axs, plot_idx, gt, gp_mean, gp_std, obs_idxs, mean_vmin=None, mean_vmax=None,
                   std_vmin=None, std_vmax=None, diff_vmin=None, diff_vmax=None):
    gtim = axs[plot_idx, 0].imshow(gt, vmin=mean_vmin, vmax=mean_vmax, origin="lower", interpolation="none", rasterized=True)
    divider = make_axes_locatable(axs[plot_idx, 0])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(gtim, cax=cax, location="right")
    ptim = axs[plot_idx, 1].imshow(gp_mean, vmin=mean_vmin, vmax=mean_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[plot_idx, 1].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    divider = make_axes_locatable(axs[plot_idx, 1])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(ptim, cax=cax, location="right")
    ptstdim = axs[plot_idx, 2].imshow(gp_std, vmin=std_vmin, vmax=std_vmax, origin="lower", interpolation="none", rasterized=True)
    axs[plot_idx, 2].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    divider = make_axes_locatable(axs[plot_idx, 2])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    diffim = axs[plot_idx, 3].imshow(gt - gp_mean, vmin=diff_vmin, vmax=diff_vmax, origin="lower", interpolation="none", rasterized=True)
    plt.colorbar(ptstdim, cax=cax, location="right")
    axs[plot_idx, 3].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    divider = make_axes_locatable(axs[plot_idx, 3])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(diffim, cax=cax, location="right")

def init_gp_1d_plot(num_runs):
    fig, axs = plt.subplots(num_runs, 4, figsize=(8, num_runs*2.5))
    for i in range(num_runs):
        axs[i, 0].set_title('ground truth')
        axs[i, 0].set_xlabel('time')
        axs[i, 0].set_ylabel('$\\theta$')
        axs[i, 1].set_title('mean')
        axs[i, 1].set_xlabel('time')
        axs[i, 1].set_ylabel('$\\theta$')
        axs[i, 2].set_title('standard deviation')
        axs[i, 2].set_xlabel('time')
        axs[i, 2].set_ylabel('$\\theta$')
        axs[i, 3].set_title('diff')
        axs[i, 3].set_xlabel('time')
        axs[i, 3].set_ylabel('$\\theta$')
    return fig, axs

def add_gp_1d_plot(fig, axs, plot_idx, gt, gp_mean, gp_std, obs_idxs, obs_vals, mean_vmin=None, mean_vmax=None,
                   std_vmin=None, std_vmax=None, diff_vmin=None, diff_vmax=None):
    axs[plot_idx, 0].plot(gt[0])
    axs[plot_idx, 1].plot(gp_mean[0])
    axs[plot_idx, 2].plot(gp_std[0])
    axs[plot_idx, 3].plot(gt[0] - gp_mean[0])
    axs[plot_idx, 1].scatter(obs_idxs[:,1], obs_vals, c='r', marker='x')
    axs[plot_idx, 2].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')
    axs[plot_idx, 3].scatter(obs_idxs[:,1], obs_idxs[:,0], c='r', marker='x')

def animate_1d(x, u, posterior_mean_1, posterior_std_1, posterior_mean_2,
               posterior_std_2, dt, output_filename):    
    t_steps = u.shape[0]
    # Create animation
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    line_gt1, = ax[0].plot(x, u[0], label='ground truth')
    line_gt2, = ax[1].plot(x, u[0], label='ground truth')
    line_mean, = ax[0].plot(x, posterior_mean_1[0], label='posterior mean')
    line_std = ax[0].fill_between(x, posterior_mean_1[0] - posterior_std_1[0], posterior_mean_1[0] + posterior_std_1[0], facecolor='orange', alpha=0.2, label='posterior std.')
    line_mean_rbf, = ax[1].plot(x, posterior_mean_2[0], c='green', label='posterior mean (rbf)')
    line_std_rbf = ax[1].fill_between(x, posterior_mean_2[0] - posterior_std_2[0], posterior_mean_2[0] + posterior_std_2[0], facecolor='green', alpha=0.2, label='posterior std. (rbf)')
    ax[0].set_ylim([-0.1,1.2])
    ax[1].set_ylim([-0.1,1.2])
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[0].set_ylabel("u(x, t)")
    ax[1].set_ylabel("u(x, t)")
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

    def animate(i):
        line_gt1.set_ydata(u[i])
        line_gt2.set_ydata(u[i])
        line_mean.set_ydata(posterior_mean_1[i])
        line_mean_rbf.set_ydata(posterior_mean_2[i])
        ax[0].collections.clear()
        ax[1].collections.clear()
        line_std = ax[0].fill_between(x, posterior_mean_1[i] - posterior_std_1[i], posterior_mean_1[i] + posterior_std_1[i], facecolor='orange', alpha=0.2)
        line_std_rbf = ax[1].fill_between(x, posterior_mean_2[i] - posterior_std_2[i], posterior_mean_2[i] + posterior_std_2[i], facecolor='green', alpha=0.2)
        ax[0].set_title(f'SPDE Kernel t={(dt*i):02.3f}')
        ax[1].set_title(f'RBF Kernel t={(dt*i):02.3f}')
        return line_gt1, line_gt2, line_mean, line_std, line_mean_rbf, line_std_rbf

    anim = animation.FuncAnimation(fig, animate, frames=t_steps//2, interval=300, blit=True)
    anim.save(output_filename, writer='pillow', fps=2)
    return HTML(anim.to_jshtml())

def animate_images(images, obs_idxs, output_filename, fps=10):
    fig = plt.figure()  # Create a figure object

    # Create an imshow plot
    img = plt.imshow(images[0], vmin=0, vmax=1, origin="lower", extent=(0, 8, 0, 10))
    plt.scatter(obs_idxs[:,1] / 10, obs_idxs[:,0] / 10, c='r', marker='x')
    plt.xlabel('t')
    plt.ylabel('x')

    # Define the animate function
    def animate(i):
        img.set_data(images[i])  # Update image data
        # img.set_clim(vmax=images[i].max())
        plt.title(f"Iteration {i:d}")
        return img,

    t_steps = len(images)  # Number of frames

    # Create an animation
    anim = animation.FuncAnimation(fig, animate, frames=t_steps, interval=10, blit=True)

    # Save the animation
    anim.save(output_filename, writer='pillow', fps=fps)

    # Return a JSHTML representation of the animation
    return HTML(anim.to_jshtml())
