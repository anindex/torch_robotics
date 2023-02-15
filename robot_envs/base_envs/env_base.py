import sys
from abc import abstractmethod

import torch


class EnvBase:

    def __init__(self,
                 name='2d',
                 q_n_dofs=2,
                 q_min=None,
                 q_max=None,
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

    def sample_q(self, without_collision=True):
        if without_collision:
            return self.random_coll_free_q()
        else:
            return self.random_q()

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
            in_collision = self.check_collision(qs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
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
    def check_collision(self, q, **kwargs):
        raise NotImplementedError

    @staticmethod
    def distance_q(q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    @abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def render_physics(self, **kwargs):
        raise NotImplementedError

