from matplotlib import pyplot as plt

from torch_kinematics_tree.geometrics.utils import to_numpy


def plot_trajectories(ax, trajs, line_color='red', plot_markers=False, label='', plot_velocities=False, **kwargs):
    plot_options = dict(linewidth=1.)
    if trajs is not None:
        for traj in trajs:
            traj = to_numpy(traj)
            start_state = traj[0]
            goal_state = traj[-1]
            if ax.name == '3d':
                ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], marker='o', markersize=2., color=line_color, zorder=20, **plot_options)
                ax.scatter3D(start_state[0], start_state[1], start_state[2], color='g', marker='o', zorder=20, s=100)
                ax.scatter3D(goal_state[0], goal_state[1], goal_state[2], color='r', marker='o', zorder=20, s=100)
            else:
                ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=2., color=line_color, zorder=20, **plot_options)
                ax.scatter(start_state[0], start_state[1],  color='g', marker='o', s=100, zorder=20)
                ax.scatter(goal_state[0], goal_state[1],  color='r', marker='o', s=100, zorder=20)

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


def plot_trajectories_in_time(trajs, q_n_dofs=2, ax=None, **plot_options):
    if trajs is not None:
        if ax is None:
            if trajs[0].shape[-1] > q_n_dofs:
                fig, axs = plt.subplots(q_n_dofs, 2, squeeze=False, figsize=(12, 1.5*q_n_dofs))
                axs[0, 0].set_title('Position')
                axs[0, 1].set_title('Velocity')
            else:
                fig, axs = plt.subplots(q_n_dofs, 1, squeeze=False, figsize=(12, 1.5*q_n_dofs))
                axs[0, 0].set_title('Position')
        else:
            axs = ax

        for traj in trajs:
            traj = to_numpy(traj)
            traj_pos = traj[:, :q_n_dofs]
            for i, q_pos in enumerate(traj_pos.T):
                axs[i, 0].plot(q_pos, **plot_options)
                axs[i, 0].set_ylabel(f'q{i}')
            axs[-1, 0].set_xlabel('Step')

            if trajs[0].shape[-1] > q_n_dofs:
                traj_vel = traj[:, q_n_dofs:]
                for i, q_vel in enumerate(traj_vel.T):
                    axs[i, 1].plot(q_vel, **plot_options)
                    axs[i, 1].set_ylabel(f'dq{i}')
                axs[-1, 1].set_xlabel('Step')

        if ax is None:
            fig.tight_layout()
            return fig, axs

