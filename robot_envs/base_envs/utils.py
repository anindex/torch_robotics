from matplotlib import pyplot as plt

from torch_kinematics_tree.geometrics.utils import to_numpy


def plot_trajectories(ax, trajs, color='red', plot_markers=False, label='', plot_velocities=False,
                      linewidth=1.0, s=100, markersize=2,
                      **kwargs):
    if trajs is not None:
        for traj in trajs:
            traj = to_numpy(traj)
            start_state = traj[0]
            goal_state = traj[-1]
            if ax.name == '3d':
                ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], marker='o', markersize=markersize, color=color, zorder=1, linewidth=linewidth)
                ax.scatter3D(start_state[0], start_state[1], start_state[2], color='green', marker='o', zorder=1, s=s)
                ax.scatter3D(goal_state[0], goal_state[1], goal_state[2], color='blue', marker='o', zorder=1, s=s)
            else:
                zorder = kwargs.get('zorder', 1)
                ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=markersize, color=color, zorder=zorder, linewidth=linewidth)
                ax.scatter(start_state[0], start_state[1],  color='green', marker='o', s=s, zorder=zorder)
                ax.scatter(goal_state[0], goal_state[1],  color='blue', marker='o', s=s, zorder=zorder)

        if plot_velocities:
            plot_velocities_fn(ax, trajs)


def plot_velocities_fn(ax, trajs, color='black'):
    for traj in trajs:
        q_n_dof = traj.shape[-1] // 2
        x = traj[:, 0:q_n_dof]
        x_dot = traj[:, q_n_dof:]
        if ax.name == '3d':
            ax.quiver(x[:, 0], x[:, 1], x[:, 2], x_dot[:, 0], x_dot[:, 1], x_dot[:, 2], color=color)
        else:
            ax.quiver(x[:, 0], x[:, 1], x_dot[:, 0], x_dot[:, 1], color=color, width=2e-3)

        # ax2.plot(np.arange(traj.shape[0]), x1_dot, label='x1_dot')
        # ax2.plot(np.arange(traj.shape[0]), x2_dot, label='x2_dot')


def plot_trajectories_in_time(trajs, q_n_dofs=2, ax=None, color='black', label='gt', zorder=1, linewidth=1.,
                              limits=None,
                              eps=1e-1,
                              **plot_options):
    if trajs is not None:
        if ax is None:
            if trajs[0].shape[-1] > q_n_dofs:
                fig, axs = plt.subplots(q_n_dofs, 2, squeeze=False, figsize=(12, 2*round(1.5*q_n_dofs/2)))
                axs[0, 0].set_title('Position')
                axs[0, 1].set_title('Velocity')
            else:
                fig, axs = plt.subplots(q_n_dofs, 1, squeeze=False, figsize=(12, 2*round(1.5*q_n_dofs/2)))
                axs[0, 0].set_title('Position')
        else:
            axs = ax

        for traj in trajs:
            traj = to_numpy(traj)
            # position
            traj_pos = traj[:, :q_n_dofs]
            for i, q_pos in enumerate(traj_pos.T):
                axs[i, 0].plot(q_pos, color=color, label=label, zorder=zorder, linewidth=linewidth)
                axs[i, 0].set_ylabel(f'q{i}')
                if limits is not None:
                    limits_min = to_numpy(limits[0])
                    limits_max = to_numpy(limits[1])
                    axs[i, 0].set_ylim(limits_min[i] - eps, limits_max[i] + eps)
            axs[-1, 0].set_xlabel('Step')
            # velocity
            if trajs[0].shape[-1] > q_n_dofs:
                traj_vel = traj[:, q_n_dofs:]
                for i, q_vel in enumerate(traj_vel.T):
                    axs[i, 1].plot(q_vel, color=color, label=label, zorder=zorder, linewidth=linewidth)
                    axs[i, 1].set_ylabel(f'dq{i}')
                axs[-1, 1].set_xlabel('Step')

        if ax is None:
            fig.tight_layout()
            return fig, axs

