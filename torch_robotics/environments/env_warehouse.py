from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments import create_table_object_field, create_shelf_field
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvWarehouse(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        # table object field
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table_obj_field.fields[0].sizes[0]
        dist_robot_to_table = 0.10
        theta = np.deg2rad(90)
        ori = np.eye(3)
        ori[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        table_obj_field.set_position_orientation(
            pos=np.array([dist_robot_to_table + table_sizes[1].item()/2, 0, -table_sizes[2].item()/2]),
            ori=ori
        )

        # shelf object field
        shelf_obj_field = create_shelf_field(tensor_args=tensor_args)
        # theta = np.deg2rad(-90)
        dist_table_shelf = 0.15
        shelf_obj_field.set_position_orientation(
            pos=(dist_robot_to_table, dist_table_shelf + table_sizes[0].item()/2, -table_sizes[2].item()),
            # ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        # small shelf field
        small_shelf_sizes = [0.8, 0.28, 1.06]
        small_shelf_object = MultiBoxField(
            np.array([[0., 0., 0.]]),
            np.array([small_shelf_sizes]),
            tensor_args=tensor_args
        )
        small_shelf_object_field = ObjectField([small_shelf_object], 'small_shelf')
        small_shelf_object_field.set_position_orientation(
            pos=(dist_robot_to_table + small_shelf_sizes[0]/2, -dist_table_shelf - table_sizes[1].item()/2,
                 -small_shelf_sizes[2]/2 + (small_shelf_sizes[2]-table_sizes[2].item())),
            # ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        obj_list = [table_obj_field, shelf_obj_field, small_shelf_object_field]

        super().__init__(
            limits=torch.tensor([[-1, -1, -1], [1.5, 1., 1.5]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/80,
            n_radius=torch.pi/4,
            n_pre_samples=50000,
            max_time=15
        )
        if isinstance(robot, RobotPanda):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvWarehouse(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    plt.show()
