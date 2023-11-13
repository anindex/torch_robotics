import os.path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

import matplotlib.collections as mcoll


class RobotPointMass2D(RobotBase):

    def __init__(self,
                 q_limits=torch.tensor([[-1, -1], [1, 1]]),  # configuration space limits
                 robot_urdf_path=os.path.join(get_robot_path(), "point_mass_robot_2d.urdf"),
                 **kwargs):
        super().__init__(
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['robot'],
            link_margins_for_object_collision_checking=[0.02],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            use_collision_spheres=False,
            robot_urdf_path=robot_urdf_path,
            robot_urdf_path_ompl=robot_urdf_path,
            link_names_torchkin=['robot'],
            link_name_ee='robot',
            **kwargs
        )

    # def fk_map_collision_impl(self, q, **kwargs):
    #     # There is no forward kinematics. Assume it's the identity.
    #     # Add tasks space dimension
    #     return q.unsqueeze(-2)

    def fk_map_collision_impl(self, q, **kwargs):
        q_original_shape = q.shape
        if len(q_original_shape) == 1:
            q = q.unsqueeze(0)  # add batch dimension
        elif len(q_original_shape) == 3:
            q = einops.rearrange(q, 'b n d -> (b n) d')
        else:
            raise NotImplementedError

        link_poses = self.robot_torchkin_fk(q)
        links_poses_th = torch.cat(link_poses)
        link_positions_th = link_pos_from_link_tensor(links_poses_th)
        task_space_positions = link_positions_th[..., :self.q_dim]  # q_dim because the point mass robot can be in 2D or 3D

        if len(q_original_shape) == 1:
            raise NotImplementedError
        elif len(q_original_shape) == 3:
            task_space_positions = einops.rearrange(task_space_positions, '(b n) d -> b n d', b=q_original_shape[0])
        else:
            raise NotImplementedError

        # Add tasks space dimension
        return task_space_positions.unsqueeze(-2)

    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        if q is not None:
            margin = self.link_margins_for_object_collision_checking[0] * margin_multiplier
            q = to_numpy(q)
            if q.ndim == 1:
                if self.q_dim == 2:
                    circle1 = plt.Circle(q, margin, color=color, zorder=10)
                    ax.add_patch(circle1)
                elif self.q_dim == 3:
                    plot_sphere(ax, q, np.zeros_like(q), margin, cmap)
                else:
                    raise NotImplementedError
            elif q.ndim == 2:
                if q.shape[-1] == 2:
                    # ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                    circ = []
                    for q_ in q:
                        circ.append(plt.Circle(q_, margin, color=color))
                        coll = mcoll.PatchCollection(circ, zorder=10)
                        ax.add_collection(coll)
                elif q.shape[-1] == 3:
                    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                    for q_ in q:
                        plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', control_points=None,
            **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            trajs_np = to_numpy(trajs_pos)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                if control_points is not None:
                    points = np.reshape(to_numpy(control_points), (-1, 2))
                    colors_scatter = ['blue'] * points.shape[0]
                    # for control_points_aux, color in zip(control_points, colors):
                    #     colors_scatter.extend([color]*control_points_aux.shape[0])
                else:
                    points = np.reshape(trajs_np, (-1, 2))
                    colors_scatter = ['red'] * points.shape[0]
                    # colors_scatter = []
                    # for segment, color in zip(segments, colors):
                    #     colors_scatter.extend([color]*segment.shape[0])
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
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o', color='purple', markersize=7)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=7)


# alias for backward compatilibity
RobotPointMass = RobotPointMass2D


class RobotPointMass3D(RobotPointMass2D):

    def __init__(self, **kwargs):
        super().__init__(
            q_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **kwargs['tensor_args']),  # configuration space limits
            robot_urdf_path=os.path.join(get_robot_path(), "point_mass_robot_3d.urdf"),
            **kwargs
        )

