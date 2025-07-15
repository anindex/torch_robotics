import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField, MultiSphereField
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, get_torch_device
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

class EnvCrazyflie3D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        obj_list = [
            MultiBoxField(
                np.array(
                [[0, 0, 0],
                 ]
                ),
                np.array(
                [[0.1, -0.1, 0.1]
                 ]
                )
                ,
                tensor_args=tensor_args
                )
        ]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[0, -3, 0], [2, 0, 2]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, 'square2d')],
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=30
        )
        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=100,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

