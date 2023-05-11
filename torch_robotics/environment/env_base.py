import sys
from abc import abstractmethod, ABC

import numpy as np
import torch

from torch_robotics.environment.obst_map import OccupancyMap


class EnvBase(ABC):

    def __init__(self,
                 name='NameEnvBase',
                 limits=None,
                 obj_list=None,
                 tensor_args=None,
                 ):
        self.tensor_args = tensor_args

        self.name = name

        # Workspace
        self.limits = limits
        self.dim = len(limits[0])

        ################################################################################################
        # Objects
        self.obj_list = obj_list
        self.simplify_primitives()

        ################################################################################################
        # Occupancy map
        self.occupancy_map = None
        self.cell_size = None

    def simplify_primitives(self):
        return
        # Groups primitives of the same type for faster batched computation
        raise NotImplementedError

    def build_occupancy_map(self, cell_size=None):
        self.cell_size = cell_size
        # Make occupancy grid
        map_dim = torch.abs(self.limits[1] - self.limits[0])
        occ_map = OccupancyMap(map_dim, self.cell_size, tensor_args=self.tensor_args)
        for obj in self.obj_list:
            obj.add_to_occupancy_map(occ_map)
        occ_map.convert_map_to_torch()
        self.occupancy_map = occ_map

    def add_obstacle_primitive(self, obst_primitive):
        raise NotImplementedError
        self.obst_primitives_l.append(obst_primitive)
        self.simplify_primitives()

    def zero_grad(self):
        for obj in self.obj_list:
            try:
                obj.zero_grad()
            except:
                raise NotImplementedError

    def render(self, ax=None):
        for obj in self.obj_list:
            obj.render(ax)

        # if self.use_occupancy_map:
        #     res = self.occupancy_map.map.shape[0]
        #     x = np.linspace(self.limits[0][0], self.limits[1][0], res)
        #     y = np.linspace(self.limits[0][1], self.limits[1][1], res)
        #     map = self.occupancy_map.map
        #     map[map > 1] = 1
        #     ax.contourf(x, y, map, 2, cmap='Greys')
        #     ax.set_aspect('equal')
        #     ax.set_facecolor('white')

    def get_rrt_params(self):
        raise NotImplementedError

    def get_sgpmp_params(self):
        raise NotImplementedError

    def get_gpmp_params(self):
        raise NotImplementedError