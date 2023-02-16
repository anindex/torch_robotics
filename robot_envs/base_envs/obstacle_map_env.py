import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_envs.base_envs.env_base import EnvBase


class ObstacleMapEnv(EnvBase):

    def __init__(self,
                 name='NA',
                 q_n_dofs=2,
                 q_min=None,
                 q_max=None,
                 obstacle_map=None,
                 tensor_args=None,
                 ):
        ################################################################################################
        # Robot
        super().__init__(name=name, q_n_dofs=q_n_dofs, q_min=q_min, q_max=q_max, tensor_args=tensor_args)

        ################################################################################################
        # Obstacle map
        self.obstacle_map = obstacle_map

    def fk_map(self, qs):
        # if there is no forward kinematics, assume it's the identity
        return qs

    def _compute_collision(self, qs, **kwargs):
        if qs.ndim == 1:
            qs = qs.unsqueeze(0).unsqueeze(0)  # add batch and trajectory_length dimension for interface
        elif qs.ndim == 2:
            qs = qs.unsqueeze(1)  # add trajectory_length dimension for interface

        collisions = torch.ones(qs.shape[0], **self.tensor_args)

        ########################################
        # Configuration space Boundaries
        idxs_in_bounds = torch.argwhere(torch.all(
            torch.logical_and(torch.greater_equal(qs[:, 0, :], self.q_min),
                              torch.less_equal(qs[:, 0, :], self.q_max)
                              ), dim=-1)
        )
        idxs_in_bounds = idxs_in_bounds.squeeze()
        collisions[idxs_in_bounds] = 0

        # check if all points are out of bounds (in collision)
        if torch.count_nonzero(collisions) == qs.shape[0]:
            return collisions

        ########################################
        # Task space collisions
        # do forward kinematics
        if idxs_in_bounds.ndim == 0:
            qs_try = qs[idxs_in_bounds][None, ...]
        else:
            qs_try = qs[idxs_in_bounds]

        pos_x = self.fk_map(qs_try)

        # collision in task space
        collisions_task_space = torch.zeros_like(collisions[idxs_in_bounds])
        collisions_pos_x = self.obstacle_map.get_collisions(pos_x, **kwargs).squeeze()
        if collisions_pos_x.ndim == 2:
            # configuration is not valid if any point in the task space is in collision
            idx_collisions = torch.argwhere(torch.any(collisions_pos_x, dim=-1)).squeeze()
        else:
            idx_collisions = torch.argwhere(collisions_pos_x)

        if idx_collisions.nelement() > 0:
            collisions_task_space[idx_collisions] = 1

        # filter collisions in task space
        collisions = torch.logical_or(collisions[idxs_in_bounds], collisions_task_space)

        return collisions

    def render(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        res = self.obstacle_map.map.shape[0]
        x = np.linspace(self.q_min[0], self.q_min[1], res)
        y = np.linspace(self.q_min[0], self.q_max[1], res)
        map = self.obstacle_map.map
        map[map > 1] = 1
        ax.contourf(x, y, map, 2, cmap='Greys')

        ax.set_aspect('equal')
        ax.set_facecolor('white')
