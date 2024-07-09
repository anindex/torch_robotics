import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments import EnvPlanar2Link
from torch_robotics.robots import RobotPlanar2Link
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, get_torch_device, to_numpy

from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def plot_configuration_free_space(
        env,
        robot,
        task,
        fig=None,
        ax=None
):
    if fig is None and ax is None:
        fig, ax = create_fig_and_axes(2)

    q_limits_low = robot.q_min
    q_limits_high = robot.q_max

    # create meshgrid
    N = 600
    q1 = torch.linspace(q_limits_low[0], q_limits_high[0], N, **robot.tensor_args)
    q2 = torch.linspace(q_limits_low[1], q_limits_high[1], N, **robot.tensor_args)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing='ij')

    qs_flat = torch.stack([Q1.flatten(), Q2.flatten()], dim=-1)

    Z = task.compute_collision(qs_flat, margin=0)
    valid = Z.reshape(Q1.shape)

    # plot the meshgrid
    cMap = matplotlib.colors.ListedColormap(['white', 'grey'])
    ax.contourf(to_numpy(Q1), to_numpy(Q2), to_numpy(valid), cmap=cMap)

    ax.set_xlabel('$q_1$ [rad]')
    ax.set_ylabel('$q_2$ [rad]')

    return fig, ax


if __name__ == '__main__':
    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    tensor_args = DEFAULT_TENSOR_ARGS

    env = EnvPlanar2Link(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.001,
        tensor_args=tensor_args
    )

    robot = RobotPlanar2Link(
        tensor_args=tensor_args
    )

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1., -1.], [1., 1.]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.01,
        tensor_args=tensor_args
    )
    fig, ax = plot_configuration_free_space(env, robot, task)
    fig.tight_layout()
    plt.show()
