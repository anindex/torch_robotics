import os

import numpy as np
import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from torch_robotics.torch_utils.torch_utils import to_numpy, tensor_linspace_v1, to_torch


class RobotPlanar2Link(RobotBase):

    link_name_ee = 'link_ee'  # must be in the urdf file

    def __init__(self,
                 urdf_robot_file=os.path.join(get_robot_path(), "planar_robot_2_link.urdf"),
                 collision_spheres_file_path=os.path.join(
                     get_configs_path(), 'planar_robot_2_link/planar_robot_2_link_sphere_config.yaml'),
                 task_space_dim=2,
                 **kwargs):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            link_name_ee=self.link_name_ee,
            task_space_dim=task_space_dim,
            **kwargs
        )

        ################################################################################################
        # Robot
        self.l1 = 0.2
        self.l2 = 0.4

    def link_positions(self, q):
        pos_link0 = torch.zeros((*q.shape[0:-1], 2), **self.tensor_args)
        pos_link1 = torch.zeros_like(pos_link0)
        pos_link2 = torch.zeros_like(pos_link0)

        pos_link1[..., 0] = self.l1 * torch.cos(q[..., 0])
        pos_link1[..., 1] = self.l1 * torch.sin(q[..., 0])

        pos_link2[..., 0] = pos_link1[..., 0] + self.l2 * torch.cos(q[..., 0] + q[..., 1])
        pos_link2[..., 1] = pos_link1[..., 1] + self.l2 * torch.sin(q[..., 0] + q[..., 1])

        return pos_link0, pos_link1, pos_link2

    def render(self, ax, q=None, alpha=1.0, color='blue', linewidth=2.0, **kwargs):
        # for H in self.fk_object_collision(q.unsqueeze(0)):
        #     _p = H[0, :2, 3]
        #     _p = to_numpy(_p)
        #     ax.scatter(_p[0], _p[1], color='gray', linewidth=linewidth, alpha=alpha)
        # return
        p0, p1, p2 = self.link_positions(q)
        p1, p2 = p1.squeeze(), p2.squeeze()
        l2 = to_numpy(torch.vstack((p1, p2)))
        p1, p2 = to_numpy(p1), to_numpy(p2)
        ax.plot([0, p1[0]], [0, p1[1]], color=color, linewidth=linewidth, alpha=alpha)
        ax.plot(l2[:, 0], l2[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter(p2[0], p2[1], color='red', marker='o')

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for _trajs_pos in trajs_pos:
                for q, color in zip(_trajs_pos, colors):
                    self.render(ax, q, alpha=0.8, color=color)
        if start_state is not None:
            self.render(ax, start_state, alpha=1.0, color='blue')
        if goal_state is not None:
            self.render(ax, goal_state, alpha=1.0, color='red')
