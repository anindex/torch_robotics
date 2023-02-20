import sys
from abc import abstractmethod

import torch


class EnvBase:

    def __init__(self,
                 name='2d',
                 q_n_dofs=2,
                 q_min=None,
                 q_max=None,
                 work_space_dim=2,
                 tensor_args=None
                 ):
        self.tensor_args = tensor_args

        ################################################################################################
        self.name = name
        self.q_n_dofs = q_n_dofs

        ################################################################################################
        # Configuration space
        assert q_min is not None and q_max is not None, "q_min and q_max cannot be None"
        self.q_min = q_min.to(**self.tensor_args)
        self.q_max = q_max.to(**self.tensor_args)
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)

        ################################################################################################
        # Work space
        self.work_space_dim = work_space_dim

    def sample_q(self, without_collision=True, **kwargs):
        if without_collision:
            return self.random_coll_free_q(**kwargs)
        else:
            return self.random_q(**kwargs)

    def random_q(self, n_samples=1):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def random_coll_free_q(self, n_samples=1, max_samples=1000):
        # Random position in configuration space collision free
        max_tries = 1000
        reject = True
        samples = torch.zeros((n_samples, self.q_n_dofs), **self.tensor_args)
        idx_begin = 0
        for i in range(max_tries):
            qs = self.random_q(max_samples)
            in_collision = self._compute_collision(qs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            if idxs_not_in_collision.nelement() == 1:
                idxs_not_in_collision = [idxs_not_in_collision]
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()

    @abstractmethod
    def _compute_collision(self, q, **kwargs):
        raise NotImplementedError

    def compute_collision(self, q, **kwargs):
        q_pos = self.get_q_position(q)
        return self._compute_collision(q_pos, **kwargs)

    @abstractmethod
    def _compute_collision_cost(self, q, **kwargs):
        raise NotImplementedError

    def compute_collision_cost(self, q, **kwargs):
        q_pos = self.get_q_position(q)
        return self._compute_collision_cost(q_pos, **kwargs)

    def get_q_position(self, q):
        return q[..., :self.q_n_dofs]

    def get_q_velocity(self, q):
        return q[..., self.q_n_dofs:2*self.q_n_dofs]

    @staticmethod
    def distance_q(q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    @abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render_trajectories(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render_physics(self, **kwargs):
        raise NotImplementedError

    def get_rrt_params(self):
        raise NotImplementedError

    def get_sgpmp_params(self):
        raise NotImplementedError

