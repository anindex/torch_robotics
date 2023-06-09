import numpy as np
import torch

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, tensor_linspace


class RobotPlanar2Link(RobotBase):

    def __init__(self,
                 tensor_args=None,
                 **kwargs):

        q_limits = torch.tensor([[-torch.pi, -torch.pi + 0.01], [torch.pi, torch.pi - 0.01]], **tensor_args)

        super().__init__(
            q_limits=q_limits,
            tensor_args=tensor_args,
            **kwargs
        )

        ################################################################################################
        # Robot
        self.l1 = 0.2
        self.l2 = 0.4

    def end_link_positions(self, q):
        pos_end_link1 = torch.zeros((*q.shape[0:-1], 2), **self.tensor_args)
        pos_end_link2 = torch.zeros((*q.shape[0:-1], 2), **self.tensor_args)

        pos_end_link1[..., 0] = self.l1 * torch.cos(q[..., 0])
        pos_end_link1[..., 1] = self.l1 * torch.sin(q[..., 0])

        pos_end_link2[..., 0] = pos_end_link1[..., 0] + self.l2 * torch.cos(q[..., 0] + q[..., 1])
        pos_end_link2[..., 1] = pos_end_link1[..., 1] + self.l2 * torch.sin(q[..., 0] + q[..., 1])

        return pos_end_link1, pos_end_link2

    def fk_map_impl(self, q, pos_only=False):
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        points_along_links = 25
        p1, p2 = self.end_link_positions(q)
        positions_link1 = tensor_linspace(torch.zeros_like(p1), p1, points_along_links)
        positions_link1 = positions_link1.swapaxes(-2, -1)
        positions_link2 = tensor_linspace(p1 + self.self_collision_margin, p2, points_along_links)
        positions_link2 = positions_link2.swapaxes(-2, -1)

        x_pos = torch.cat((positions_link1, positions_link2), dim=-2)
        if pos_only:
            return x_pos
        else:
            raise NotImplementedError

    def render(self, ax, q=None, alpha=1.0, color='blue', linewidth=2.0, **kwargs):
        p1, p2 = self.end_link_positions(q)
        p1, p2 = p1.squeeze(), p2.squeeze()
        l2 = to_numpy(torch.vstack((p1, p2)))
        p1, p2 = to_numpy(p1), to_numpy(p2)
        ax.plot([0, p1[0]], [0, p1[1]], color=color, linewidth=linewidth, alpha=alpha)
        ax.plot(l2[:, 0], l2[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter(p2[0], p2[1], color='red', marker='o')

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            for q, color in zip(trajs, colors):
                q = q.view(1, -1)
                self.render(ax, q, alpha=0.8, color=color)
        if start_state is not None:
            self.render(ax, start_state.view(1, -1), alpha=1.0, color='blue')
        if goal_state is not None:
            self.render(ax, goal_state.view(1, -1), alpha=1.0, color='red')
