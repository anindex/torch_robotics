from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import jacobian


class GridMapSDF:
    """
    Generates an SDF grid.
    """
    def __init__(self, limits, cell_size, obj_list, tensor_args=None):

        self.limits = limits
        self.dim = limits.shape[-1]

        self.tensor_args = tensor_args

        # objects
        self.obj_list = obj_list

        # SDF grids
        map_dim = torch.abs(limits[1] - limits[0])
        self.map_dim = map_dim
        self.cell_size = cell_size
        self.cmap_dim = [0 for _ in map_dim]
        for i, d in enumerate(self.cmap_dim):
            self.cmap_dim[i] = ceil(map_dim[i]/cell_size)

        self.points_for_sdf = None
        self.sdf_tensor = None
        self.grad_sdf_tensor = None
        self.precompute_sdf()

        # Map center (in cells)
        self.origin = np.array([d//2 for d in self.cmap_dim])
        self.c_offset = torch.Tensor(self.origin).to(**tensor_args)

    def precompute_sdf(self):
        # create voxel grid of points
        basis_ranges = [
            torch.linspace(self.limits[0][0], self.limits[1][0], self.cmap_dim[0], **self.tensor_args),
            torch.linspace(self.limits[0][1], self.limits[1][1], self.cmap_dim[1], **self.tensor_args),
        ]
        if self.dim == 3:
            basis_ranges.append(
                torch.linspace(self.limits[0][2], self.limits[1][2], self.cmap_dim[2], **self.tensor_args)
            )
        points_for_sdf_meshgrid = torch.meshgrid(*basis_ranges, indexing='ij')
        self.points_for_sdf = torch.stack(points_for_sdf_meshgrid, dim=-1)

        # compute the sdf and its gradient for all points
        f_ravel = lambda x: x.ravel()
        points_for_sdf_flat = torch.cat([f_ravel(ps).view(-1, 1) for ps in points_for_sdf_meshgrid], dim=-1)

        sdf_tensor = self.compute_signed_distance_raw(points_for_sdf_flat)

        f_grad_sdf = lambda x: self.compute_signed_distance_raw(x).sum()
        grad_sdf_tensor = jacobian(f_grad_sdf, self.points_for_sdf)

        self.sdf_tensor = sdf_tensor.reshape(*self.cmap_dim)
        self.grad_sdf_tensor = grad_sdf_tensor.reshape((*self.cmap_dim, self.dim))

    def compute_signed_distance_raw(self, x):
        sdf = None
        for obj in self.obj_list:
            sdf_obj = obj.compute_signed_distance(x)
            if sdf is None:
                sdf = sdf_obj
            else:
                sdf = torch.minimum(sdf, sdf_obj)
        return sdf

    def __call__(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def compute_cost(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def compute_signed_distance(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def get_sdf(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, horizon, task_spaces, position_dim)
        :return: collision cost on the trajectories
        """
        X_in_map = X * (1/self.cell_size) + self.c_offset
        X_in_map = X_in_map.floor()

        X_in_map = X_in_map.type(torch.LongTensor)

        # Project out-of-bounds locations to axis
        for i in range(X_in_map.shape[-1]):
            X_in_map[..., i] = X_in_map[..., i].clamp(0, self.points_for_sdf.shape[i]-1)

        # SDFs and gradients
        # To compute the gradients, because we already have computed the gradients, we use a surrogate sdf function
        # surrogate_sdf(x) = sdf(x_detachted) + x @ grad_sdf(x_detachted) - x_detached @ grad_sdf(x_detachted)
        # This surrogate has the property
        # surrogate_sdf(x) = sdf(x_detachted)
        # grad_x_surrogate_sdf(x) = grad_sdf(x_detachted)
        try:
            X_in_map_detached = X_in_map.detach()
            grad_sdf = 0.
            if self.dim == 2:
                sdf_vals = self.sdf_tensor[X_in_map_detached[..., 1], X_in_map_detached[..., 0]]
                if X.requires_grad:
                    grad_sdf = self.grad_sdf_tensor[X_in_map_detached[..., 1], X_in_map_detached[..., 0]]
            elif self.dim == 3:
                sdf_vals = self.sdf_tensor[X_in_map_detached[..., 1], X_in_map_detached[..., 0], X_in_map_detached[..., 2]]
                if X.requires_grad:
                    grad_sdf = self.grad_sdf_tensor[X_in_map_detached[..., 1], X_in_map_detached[..., 0], X_in_map_detached[..., 2]]
            else:
                raise NotImplementedError

            if X.requires_grad:
                X_detached = X.detach()
                sdf_vals += (X * grad_sdf).sum(-1) - (X_detached * grad_sdf).sum(-1)

        except Exception as e:
            print(e)
            print(X_in_map)

        return sdf_vals

    def zero_grad(self):
        pass
