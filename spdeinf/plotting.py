import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.core.display import HTML

def plot_gp_2d(gt, gp_mean, gp_std, obs_idx, output_filename, mean_vmin, mean_vmax,
               std_vmin, std_vmax, diff_vmin, diff_vmax):
    fig, axs = plt.subplots(1, 4, figsize=(15,5.3))
    gtim = axs[0].imshow(gt, vmin=mean_vmin, vmax=mean_vmax, origin="lower")
    axs[0].set_title('ground truth')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('x')
    fig.colorbar(gtim)
    ptim = axs[1].imshow(gp_mean, vmin=mean_vmin, vmax=mean_vmax, origin="lower")
    axs[1].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
    axs[1].set_title('mean')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('x')
    fig.colorbar(ptim)
    ptstdim = axs[2].imshow(gp_std, vmin=std_vmin, vmax=std_vmax, origin="lower")
    axs[2].scatter(obs_idx[:,1], obs_idx[:,0], c='r', marker='x')
    axs[2].set_title('standard deviation')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('x')
    fig.colorbar(ptstdim)
    diffim = axs[3].imshow(gt - gp_mean, vmin=diff_vmin, vmax=diff_vmax, origin="lower")
    axs[3].set_title('diff')
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('x')
    fig.colorbar(diffim)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

def animate_1d(x, u, posterior_mean_1, posterior_std_1, posterior_mean_2,
               posterior_std_2, dt, output_filename):    
    t_steps = u.shape[0]
    # Create animation
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
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

def animate_images(images, obs_idx, output_filename, fps=10):
    fig = plt.figure()  # Create a figure object

    # Create an imshow plot
    img = plt.imshow(images[0], vmin=0, vmax=1, origin="lower", extent=(0, 8, 0, 10))
    plt.scatter(obs_idx[:,1] / 10, obs_idx[:,0] / 10, c='r', marker='x')
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
