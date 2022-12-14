import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from abc import ABC, abstractmethod
import os.path as osp
from copy import deepcopy


class Obstacle(ABC):
    """
    Base 2D Obstacle class
    """

    def __init__(self,center_x,center_y):
        self.center_x = center_x
        self.center_y = center_y

    @abstractmethod
    def _obstacle_collision_check(self, obst_map):
        pass

    @abstractmethod
    def _point_collision_check(self, obst_map, pts):
        pass

    @abstractmethod
    def _add_to_map(self, obst_map):
        pass


class ObstacleRectangle(Obstacle):
    """
    Derived 2D rectangular Obstacle class
    """

    def __init__(
            self,
            center_x=0,
            center_y=0,
            width=None,
            height=None,
    ):
        super().__init__(center_x, center_y)
        self.width = width
        self.height = height

    def _obstacle_collision_check(self, obst_map):
        valid=True
        obst_map_test = self._add_to_map(deepcopy(obst_map))
        if (np.any( obst_map_test.map > 1)):
            valid=False
        return valid

    def _point_collision_check(self,obst_map,pts):
        valid=True
        if pts is not None:
            obst_map_test = self._add_to_map(np.copy(obst_map))
            for pt in pts:
                if (obst_map_test[ ceil(pt[0]), ceil(pt[1])] == 1):
                    valid=False
                    break
        return valid

    def _add_to_map(self, obst_map):
        # Convert dims to cell indicies
        w = ceil(self.width / obst_map.cell_size)
        h = ceil(self.height / obst_map.cell_size)
        c_x = ceil(self.center_x / obst_map.cell_size)
        c_y = ceil(self.center_y / obst_map.cell_size)

        obst_map.map[
        c_y - ceil(h/2.) + obst_map.origin_yi:
        c_y + ceil(h/2.) + obst_map.origin_yi,
        c_x - ceil(w/2.) + obst_map.origin_xi:
        c_x + ceil(w/2.) + obst_map.origin_xi,
        ] = 1
        return obst_map


class ObstacleMap:
    """
    Generates an occupancy grid.
    """
    def __init__(self, map_dim, cell_size, tensor_args=None):

        assert map_dim[0] % 2 == 0
        assert map_dim[1] % 2 == 0

        if tensor_args is None:
            tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        cmap_dim = [0,0]
        cmap_dim[0] = ceil(map_dim[0]/cell_size)
        cmap_dim[1] = ceil(map_dim[1]/cell_size)

        self.map = np.zeros(cmap_dim)
        self.cell_size = cell_size

        # Map center (in cells)
        self.origin_xi = int(cmap_dim[0]/2)
        self.origin_yi = int(cmap_dim[1]/2)

        # self.xlim = map_dim[0]

        self.x_dim, self.y_dim = self.map.shape
        x_range = self.cell_size * self.x_dim
        y_range = self.cell_size * self.y_dim
        self.xlim = [-x_range/2, x_range/2]
        self.ylim = [-y_range/2, y_range/2]

        self.c_offset = torch.Tensor([self.origin_xi, self.origin_yi]).to(**tensor_args)

    def convert_map(self):
        self.map_torch = torch.Tensor(self.map).to(**self.tensor_args)
        return self.map_torch

    def plot(self, save_dir=None, filename="obst_map.png"):
        fig = plt.figure()
        plt.imshow(self.map)
        plt.gca().invert_yaxis()
        plt.show()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, filename))
        return fig

    def get_xy_grid(self, device):
        xv, yv = torch.meshgrid([torch.linspace(self.xlim[0], self.xlim[1], self.x_dim),
                                 torch.linspace(self.ylim[0], self.ylim[1], self.y_dim)])
        xy_grid = torch.stack((xv, yv), dim=2)
        return xy_grid.to(device)

    def get_collisions(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, traj_length, position_dim)
        :return: collision cost on the trajectories
        """

        # Convert traj. positions to occupancy indicies
        # try:
        #     c_offset = torch.Tensor([self.origin_xi, self.origin_yi]).double().to(device)
        # except Exception as e:
        #     print("Exception: ", e)
        #     print("self.origin_xi", self.origin_xi)
        #     print("self.origin_yi", self.origin_yi)

        X_occ = X * (1/self.cell_size) + self.c_offset
        X_occ = X_occ.floor()

        # X_occ = X_occ.cpu().numpy().astype(np.int)
        X_occ = X_occ.type(torch.LongTensor)
        X_occ = X_occ.to(device=self.tensor_args['device'])

        # Project out-of-bounds locations to axis
        X_occ[...,0] = X_occ[...,0].clamp(0, self.map.shape[0]-1)
        X_occ[...,1] = X_occ[...,1].clamp(0, self.map.shape[1]-1)

        # Collisions
        try:
            # collision_vals = self.map_torch[X_occ[...,0],X_occ[...,1]]
            collision_vals = self.map_torch[X_occ[...,1], X_occ[...,0]]
        except Exception as e:
            print(e)
            print(X_occ)
            print(X_occ.clamp(0, self.map.shape[0]-1))
        return collision_vals

    def compute_cost(self, X, **kwargs):
        return self.get_collisions(X, **kwargs)

    def zero_grad(self):
        pass
