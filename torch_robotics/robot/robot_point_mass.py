import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy

import matplotlib.collections as mcoll


class RobotPointMass(RobotBase):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def fk_map_impl(self, q, pos_only=False):
        # There is no forward kinematics. Assume it's the identity.
        # Add task space dimension
        if pos_only:
            return q.unsqueeze(-2)
        else:
            # no rotation
            H = torch.eye(self.q_dim + 1, **self.tensor_args).unsqueeze(0).unsqueeze(-3).repeat(*q.shape[:-1], 1, 1, 1)
            H[..., :self.q_dim, self.q_dim] = q.unsqueeze(-2)
            return H

    def render(self, ax, q=None, color='blue', **kwargs):
        if q is not None:
            q = to_numpy(q)
            if q.ndim == 1:
                ax.scatter(*q, color=color, s=10**2, zorder=10)
            elif q.ndim == 2:
                if q.shape[-1] == 2:
                    ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                elif q.shape[-1] == 3:
                    ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'], **kwargs):
        if trajs is not None:
            trajs_np = to_numpy(trajs)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle='solid')
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle='solid')
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 2))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            start_state_np = to_numpy(start_state)
            if len(start_state_np) == 3:
                ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=7)
            else:
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=7)
        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            if len(goal_state_np) == 3:
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], 'ro', markersize=7)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], 'ro', markersize=7)
