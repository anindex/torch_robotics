import abc
from abc import ABC

import torch


class RobotBase(ABC):

    def __init__(
            self,
            q_limits=None,
            self_collision_margin=0.001,
            num_interpolate=4,
            link_interpolate_range=[2, 7],
            tensor_args=None,
            **kwargs
    ):
        self.tensor_args = tensor_args

        ################################################################################################
        # Configuration space
        assert q_limits is not None, "q_limits cannot be None"
        self.q_limits = q_limits
        self.q_min = q_limits[0]
        self.q_max = q_limits[1]
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)
        self.q_n_dofs = len(self.q_min)

        # Collision field
        self.self_collision_margin = self_collision_margin
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def get_position(self, x):
        return x[..., :self.q_n_dofs]

    def get_velocity(self, x):
        return x[..., self.q_n_dofs:2*self.q_n_dofs]

    def get_acceleration(self, x):
        raise NotImplementedError

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def fk_map(self, q):
        # q: (..., q_dim)
        # return: (..., taskspaces, x_dim)
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        return self.fk_map_impl(q)

    @abc.abstractmethod
    def fk_map_impl(self, q):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectory(self, ax, **kwargs):
        raise NotImplementedError
