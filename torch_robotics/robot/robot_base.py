import abc
from abc import ABC

import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


class RobotBase(ABC):

    def __init__(
            self,
            name='RobotBase',
            q_limits=None,
            grasped_object=None,
            margin_for_grasped_object_collision_checking=0.001,
            link_names_for_object_collision_checking=None,
            link_margins_for_object_collision_checking=None,
            self_collision_margin=0.001,
            num_interpolated_points_for_object_collision_checking=50,
            tensor_args=None,
            **kwargs
    ):
        self.name = name
        self.tensor_args = tensor_args

        ################################################################################################
        # Configuration space
        assert q_limits is not None, "q_limits cannot be None"
        self.q_limits = q_limits
        self.q_min = q_limits[0]
        self.q_max = q_limits[1]
        self.q_min_np = to_numpy(self.q_min)
        self.q_max_np = to_numpy(self.q_max)
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)
        self.q_dim = len(self.q_min)

        # Grasped object
        self.grasped_object = grasped_object
        self.margin_for_grasped_object_collision_checking = margin_for_grasped_object_collision_checking

        # Collision field
        assert num_interpolated_points_for_object_collision_checking >= len(link_names_for_object_collision_checking)
        self.self_collision_margin = self_collision_margin
        self.num_interpolated_points_for_object_collision_checking = num_interpolated_points_for_object_collision_checking
        self.link_names_for_object_collision_checking = link_names_for_object_collision_checking
        self.n_links_for_object_collision_checking = len(link_names_for_object_collision_checking)
        self.link_margins_for_object_collision_checking = link_margins_for_object_collision_checking
        self.link_margins_for_object_collision_checking_robot_tensor = torch.tensor(
            link_margins_for_object_collision_checking, **self.tensor_args).repeat_interleave(
            int(num_interpolated_points_for_object_collision_checking / len(link_margins_for_object_collision_checking))
        )
        self.link_margins_for_object_collision_checking_tensor = self.link_margins_for_object_collision_checking_robot_tensor
        # append grasped object margins
        if self.grasped_object is not None:
            self.link_margins_for_object_collision_checking_tensor = torch.cat(
                (self.link_margins_for_object_collision_checking_tensor,
                 torch.ones(self.grasped_object.n_base_points_for_collision, **self.tensor_args) * self.margin_for_grasped_object_collision_checking)
            )

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def get_position(self, x):
        return x[..., :self.q_dim]

    def get_velocity(self, x):
        return x[..., self.q_dim:2 * self.q_dim]

    def get_acceleration(self, x):
        raise NotImplementedError

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def fk_map_collision(self, q, **kwargs):
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        return self.fk_map_collision_impl(q, **kwargs)

    @abc.abstractmethod
    def fk_map_collision_impl(self, q, **kwargs):
        # q: (..., q_dim)
        # return: (..., links_collision_positions, 3)
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
