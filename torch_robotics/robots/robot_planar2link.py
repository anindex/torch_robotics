import numpy as np
import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, tensor_linspace_v1, to_torch


class RobotPlanar2Link(RobotBase):

    def __init__(self,
                 name='RobotPlanar2Link',
                 q_limits=torch.tensor([[-torch.pi, -torch.pi + 0.01], [torch.pi, torch.pi - 0.01]]),  # configuration space limits
                 **kwargs):
        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['link_0', 'link_1', 'link_2'],
            link_margins_for_object_collision_checking=[0.01, 0.01, 0.01],
            link_idxs_for_object_collision_checking=[0, 1, 2],
            num_interpolated_points_for_object_collision_checking=10,
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

    def fk_map_collision_impl(self, q, **kwargs):
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        points_along_links = 25
        p0, p1, p2 = self.link_positions(q)

        link_pos = torch.cat((p0, p1, p2), dim=-2)
        return link_pos

    def render(self, ax, q=None, alpha=1.0, color='blue', linewidth=2.0, **kwargs):
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
            for q, color in zip(trajs_pos, colors):
                q = q.view(1, -1)
                self.render(ax, q, alpha=0.8, color=color)
        if start_state is not None:
            self.render(ax, start_state.view(1, -1), alpha=1.0, color='blue')
        if goal_state is not None:
            self.render(ax, goal_state.view(1, -1), alpha=1.0, color='red')
