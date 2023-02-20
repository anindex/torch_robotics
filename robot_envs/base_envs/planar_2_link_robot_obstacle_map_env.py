import matplotlib.pyplot as plt
import numpy as np
import torch

from mp_baselines.planners.utils import tensor_linspace
from robot_envs.base_envs.env_base import EnvBase
from robot_envs.base_envs.obstacle_map_env import ObstacleMapEnv
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map, build_obstacle_map
from torch_planning_objectives.fields.primitive_distance_fields import Sphere


def create_circles_spaced():
    circles = [
        # (-0.2, 0., 0.05),
        (0.2, 0.5, 0.3),
        (0.3, 0.15, 0.1),
        (0.5, 0.5, 0.1),
        (0.3, -0.5, 0.2),
        (-0.5, 0.5, 0.3),
        (-0.5, -0.5, 0.3),
    ]
    circles = np.array(circles)
    primitive_obst_list = [Sphere(circles[:, :2], circles[:, 2])]
    return primitive_obst_list


class RobotPlanarTwoLink(ObstacleMapEnv):

    def __init__(self, tensor_args=None,):
        ################################################################################################
        # Obstacles
        limits = torch.tensor([[-np.pi, np.pi], [-np.pi + 0.01, np.pi - 0.01]], **tensor_args)

        obst_list = create_circles_spaced()

        cell_size = 0.01
        map_dim = (2, 2)
        obst_params = dict(
            map_dim=map_dim,
            obst_list=obst_list,
            cell_size=cell_size,
            map_type='direct',
            tensor_args=tensor_args,
        )
        obst_map = build_obstacle_map(**obst_params)

        super().__init__(
            name='planar_2_link_robot',
            q_n_dofs=2,
            q_min=limits[:, 0],
            q_max=limits[:, 1],
            work_space_dim=2,
            obstacle_map=obst_map,
            tensor_args=tensor_args
        )

        ################################################################################################
        # Robot
        self.l1 = 0.2
        self.l2 = 0.4

    def end_link_positions(self, qs):
        pos_end_link1 = torch.zeros((*qs.shape[0:2], 1, 2), **self.tensor_args)
        pos_end_link2 = torch.zeros((*qs.shape[0:2], 1, 2), **self.tensor_args)

        pos_end_link1[..., 0, 0] = self.l1 * torch.cos(qs[..., 0])
        pos_end_link1[..., 0, 1] = self.l1 * torch.sin(qs[..., 0])

        pos_end_link2[..., 0, 0] = pos_end_link1[..., 0, 0] + self.l2 * torch.cos(qs[..., 0] + qs[..., 1])
        pos_end_link2[..., 0, 1] = pos_end_link1[..., 0, 1] + self.l2 * torch.sin(qs[..., 0] + qs[..., 1])

        return pos_end_link1, pos_end_link2

    def fk_map(self, qs):
        points_along_links = 25
        p1, p2 = self.end_link_positions(qs)
        positions_link1 = tensor_linspace(torch.zeros_like(p1), p1, points_along_links)
        positions_link1 = positions_link1.swapaxes(-3, -1).squeeze(-1)
        positions_link2 = tensor_linspace(p1, p2, points_along_links)
        positions_link2 = positions_link2.swapaxes(-3, -1).squeeze(-1)

        pos_x = torch.cat((positions_link1, positions_link2), dim=-2)
        assert pos_x.ndim == 4, "batch, trajectory, points, x_dim"
        return pos_x

    def render(self, qs, ax):
        p1, p2 = map(np.squeeze, self.end_link_positions(qs))
        ax.plot([0, p1[0]], [0, p1[1]], color='blue', linewidth=1.)
        l2 = torch.vstack((p1, p2))
        ax.plot(l2[:, 0], l2[:, 1], color='blue', linewidth=1.)
        ax.scatter(p2[0], p2[1], color='red', marker='o')
