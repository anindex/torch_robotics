# MIT License

# Copyright (c) 2022 An Thai Le, João Carvalho

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# **********************************************************************
# The first version of some files were licensed as
# "Original Source License" (see below). Several enhancements and bug fixes
# were done by An Thai Le, João Carvalho since obtaining the first version.



# Original Source License:

# MIT License

# Copyright (c) 2022 Sasha Lambert

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from abc import ABC, abstractmethod
import os.path as osp
from copy import deepcopy


class Obstacle(ABC):
    """
    Base 2D, 3D Obstacle class
    """

    def __init__(self, center):
        self.center = center
        self.origin = np.array(center)
        self.n_dim = len(center)

    def _obstacle_collision_check(self, obst_map):
        valid = True
        obst_map_test = self._add_to_map(deepcopy(obst_map))
        if np.any(obst_map_test.map > 1):
            valid = False
        return valid

    def _point_collision_check(self, obst_map, pts):
        valid = True
        if pts is not None:
            obst_map_test = self._add_to_map(np.copy(obst_map))
            for pt in pts:
                if obst_map_test[ceil(pt[0]), ceil(pt[1])] >= 1:
                    valid = False
                    break
        return valid

    def get_center(self):
        return np.array([self.center_x, self.center_y])

    @abstractmethod
    def _add_to_map(self, obst_map):
        pass


class ObstacleRectangle(Obstacle):
    """
    Derived 2D/3D rectangular/box Obstacle class
    """

    def __init__(self, center, shape_dims):
        super().__init__(center)
        self.width = shape_dims[0]
        self.height = shape_dims[1]
        if self.n_dim == 3:
            self.depth = shape_dims[2]
        else:
            self.depth = None

    def _add_to_map(self, obst_map):
        # Convert dims to cell indices
        w = ceil(self.width / obst_map.cell_size)
        h = ceil(self.height / obst_map.cell_size)
        c_x = ceil(self.center[0] / obst_map.cell_size)
        c_y = ceil(self.center[1] / obst_map.cell_size)

        obst_map_origin_xi = obst_map.origin[0]
        obst_map_origin_yi = obst_map.origin[1]

        if self.n_dim == 2:
            obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi
            ] += 1
        else:
            c_z = ceil(self.center[2] / obst_map.cell_size)
            obst_map_origin_zi = obst_map.origin[2]
            d = ceil(self.depth / obst_map.cell_size)
            obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi,
                c_z - ceil(d / 2.) + obst_map_origin_zi:c_z + ceil(d / 2.) + obst_map_origin_zi
            ] += 1

        return obst_map


class ObstacleCircle(Obstacle):
    """
    Derived 2D/3D circle/sphere Obstacle class
    """

    def __init__(self, center, radius=1.):
        super().__init__(center)
        self.radius = radius

    def is_inside(self, p):
        # Check if point p is inside the discretized circle
        return np.linalg.norm(p - self.origin) <= self.radius

    def _add_to_map(self, obst_map):
        # Convert dims to cell indices
        c_r = ceil(self.radius / obst_map.cell_size)
        c_x = ceil(self.center[0] / obst_map.cell_size)
        c_y = ceil(self.center[1] / obst_map.cell_size)

        # centers in cell indices
        obst_map_origin_xi = obst_map.origin[0]
        obst_map_origin_yi = obst_map.origin[1]

        c_x_cell = c_x + obst_map_origin_xi
        c_y_cell = c_y + obst_map_origin_yi

        obst_map_x_dim = obst_map.dims[0]
        obst_map_y_dim = obst_map.dims[1]

        if self.n_dim == 3:
            c_z = ceil(self.center[2] / obst_map.cell_size)
            obst_map_origin_zi = obst_map.origin[2]
            c_z_cell = c_z + obst_map_origin_zi
            obst_map_z_dim = obst_map.dims[2]

        for i in range(c_x_cell - 2 * c_r, c_x_cell + 2 * c_r):
            if i < 0 or i >= obst_map_x_dim:
                continue
            for j in range(c_y_cell - 2 * c_r, c_y_cell + 2 * c_r):
                if j < 0 or j >= obst_map_y_dim:
                    continue

                if self.n_dim == 3:
                    for k in range(c_z_cell - 2 * c_r, c_z_cell + 2 * c_r):
                        if k < 0 or k >= obst_map_z_dim:
                            continue

                        p = np.array([(i - obst_map_origin_xi) * obst_map.cell_size,
                                      (j - obst_map_origin_yi) * obst_map.cell_size,
                                      (k - obst_map_origin_zi) * obst_map.cell_size
                                      ])

                        if self.is_inside(p):
                            obst_map.map[j, i, k] += 1

                else:
                    p = np.array([(i - obst_map_origin_xi) * obst_map.cell_size,
                                  (j - obst_map_origin_yi) * obst_map.cell_size])

                    if self.is_inside(p):
                        obst_map.map[j, i] += 1

        return obst_map


class ObstacleMap:
    """
    Generates an occupancy grid.
    """
    def __init__(self, map_dim, cell_size, tensor_args=None):

        for d in map_dim:
            assert d % 2 == 0, f"Dimension {d} is not an even number"

        self.map_dim = map_dim

        self.n_dim = len(map_dim)

        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.cmap_dim = [0 for _ in map_dim]
        for i, d in enumerate(self.cmap_dim):
            self.cmap_dim[i] = ceil(map_dim[i]/cell_size)

        self.map = np.zeros(self.cmap_dim)
        self.map_torch = None

        self.cell_size = cell_size

        # Map center (in cells)
        self.origin = np.array([d//2 for d in self.cmap_dim])
        self.c_offset = torch.Tensor(self.origin).to(**tensor_args)

        # limits
        self.dims = self.map.shape
        self.lims = [(-self.cell_size * d/2, self.cell_size * d/2) for d in self.dims]

    def convert_map(self):
        self.map_torch = torch.Tensor(self.map).to(**self.tensor_args)
        return self.map_torch

    def plot(self, ax=None, save_dir=None, filename="obst_map.png"):
        if ax is None:
            if self.n_dim == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

        if self.n_dim == 2:
            res = self.map.shape[0]
            x = np.linspace(self.lims[0][0], self.lims[0][1], res)
            y = np.linspace(self.lims[1][0], self.lims[1][1], res)
            ax.contourf(x, y, np.clip(self.map, 0, 1), 2, cmap='Greys')
        else:
            x, y, z = np.indices(np.array(self.map.shape) + 1, dtype=float)
            x -= self.origin[0]
            x = x * self.cell_size
            y -= self.origin[1]
            y = y * self.cell_size
            z -= self.origin[2]
            z = z * self.cell_size
            ax.voxels(y, x, z, self.map, facecolors='gray', edgecolor='black', shade=False, alpha=0.05)

    def get_collisions(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, traj_length, position_dim)
        :return: collision cost on the trajectories
        """

        X_occ = X * (1/self.cell_size) + self.c_offset
        X_occ = X_occ.floor()

        # X_occ = X_occ.cpu().numpy().astype(np.int)
        X_occ = X_occ.type(torch.LongTensor)
        X_occ = X_occ.to(device=self.tensor_args['device'])

        # Project out-of-bounds locations to axis
        for i in range(X_occ.shape[-1]):
            X_occ[..., i] = X_occ[..., i].clamp(0, self.map.shape[i]-1)

        # Collisions
        try:
            if X_occ.shape[-1] == 2:
                collision_vals = self.map_torch[X_occ[..., 1], X_occ[..., 0]]
            elif X_occ.shape[-1] == 3:
                collision_vals = self.map_torch[X_occ[..., 1], X_occ[..., 0], X_occ[..., 2]]
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            print(X_occ)
            print(X_occ.clamp(0, self.map.shape[0]-1))
        return collision_vals

    def compute_cost(self, X, **kwargs):
        return self.get_collisions(X, **kwargs)

    def zero_grad(self):
        pass
