import itertools
import sys
from abc import abstractmethod, ABC

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.grid_map_sdf import GridMapSDF
from torch_robotics.environment.occupancy_map import OccupancyMap
from torch_robotics.torch_utils.torch_timer import Timer
from torch_robotics.torch_utils.torch_utils import to_numpy


class EnvBase(ABC):

    def __init__(self,
                 name='NameEnvBase',
                 limits=None,
                 obj_fixed_list=None,
                 obj_movable_list=None,
                 precompute_sdf_obj_fixed=False,
                 sdf_cell_size=0.005,
                 tensor_args=None,
                 **kwargs
                 ):
        self.tensor_args = tensor_args

        self.name = name

        # Workspace
        assert limits is not None
        self.limits = limits
        self.limits_np = to_numpy(limits)
        self.dim = len(limits[0])

        ################################################################################################
        # Objects
        self.obj_fixed_list = obj_fixed_list
        self.obj_movable_list = obj_movable_list
        self.obj_all_list = set(itertools.chain.from_iterable((
            self.obj_fixed_list if self.obj_fixed_list is not None else [],
            self.obj_movable_list if self.obj_movable_list is not None else [])
        ))
        self.simplify_primitives()

        ################################################################################################
        # Precompute the SDF map of fixed objects
        self.grid_map_sdf_obj_fixed = None
        if precompute_sdf_obj_fixed:
            with Timer() as t:
                # Compute SDF grid
                self.grid_map_sdf_obj_fixed = GridMapSDF(
                    self.limits, sdf_cell_size, self.obj_fixed_list, tensor_args=self.tensor_args
                )
            print(f'Precomputing the SDF grid and gradients took: {t.elapsed:.3f} sec')

        ################################################################################################
        # Occupancy map
        self.occupancy_map = None
        self.cell_size = None

    def get_obj_list(self):
        return self.obj_all_list

    def get_df_obj_list(self):
        df_obj_l = []
        # fixed objects
        if self.grid_map_sdf_obj_fixed is not None:
            df_obj_l.append(self.grid_map_sdf_obj_fixed)
        else:
            df_obj_l.append(*self.obj_fixed_list)

        # movable objects
        if self.obj_movable_list is not None:
            df_obj_l.append(*self.obj_movable_list)

        return df_obj_l

    def add_obj(self, obj):
        # Adds an object to the environment
        raise NotImplementedError
        self.simplify_primitives()

    def simplify_primitives(self):
        return
        # Groups primitives of the same type for faster batched computation
        raise NotImplementedError

    def build_occupancy_map(self, cell_size=None):
        self.cell_size = cell_size
        # Make occupancy grid
        map_dim = torch.abs(self.limits[1] - self.limits[0])
        occ_map = OccupancyMap(map_dim, self.cell_size, tensor_args=self.tensor_args)
        for obj in self.obj_all_list:
            obj.add_to_occupancy_map(occ_map)
        occ_map.convert_map_to_torch()
        self.occupancy_map = occ_map

    def add_obstacle_primitive(self, obst_primitive):
        raise NotImplementedError
        self.obst_primitives_l.append(obst_primitive)
        self.simplify_primitives()

    def zero_grad(self):
        for obj in self.obj_all_list:
            try:
                obj.zero_grad()
            except:
                raise NotImplementedError

    def render(self, ax=None):
        if self.obj_fixed_list is not None:
            for obj in self.obj_fixed_list:
                obj.render(ax)

        if self.obj_movable_list is not None:
            for obj in self.obj_movable_list:
                obj.render(ax, color='red')

        ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
        ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
        if self.dim == 3:
            ax.set_zlim(self.limits_np[0][2], self.limits_np[1][2])
        ax.set_aspect('equal')

        # if self.use_occupancy_map:
        #     res = self.occupancy_map.map.shape[0]
        #     x = np.linspace(self.limits[0][0], self.limits[1][0], res)
        #     y = np.linspace(self.limits[0][1], self.limits[1][1], res)
        #     map = self.occupancy_map.map
        #     map[map > 1] = 1
        #     ax.contourf(x, y, map, 2, cmap='Greys')
        #     ax.set_aspect('equal')
        #     ax.set_facecolor('white')

    def compute_sdf(self, x, reshape_shape=None):
        # compute sdf of fixed objects
        sdf = None
        if self.grid_map_sdf_obj_fixed is not None:
            sdf = self.grid_map_sdf_obj_fixed(x)
            if reshape_shape:
                sdf = sdf.reshape(reshape_shape)
        else:
            for obj in self.obj_fixed_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape is not None:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                if sdf is None:
                    sdf = sdf_obj
                else:
                    sdf = torch.minimum(sdf, sdf_obj)

        # compute sdf of movable objects
        if self.obj_movable_list is not None:
            for obj in self.obj_movable_list:
                sdf_obj = obj.compute_signed_distance(x)
                if reshape_shape is not None:
                    sdf_obj = sdf_obj.reshape(reshape_shape)
                if sdf is None:
                    sdf = sdf_obj
                else:
                    sdf = torch.minimum(sdf, sdf_obj)

        return sdf

    def render_sdf(self, ax=None, fig=None):
        if self.dim == 3:
            raise NotImplementedError
        # draw sdf
        xs = torch.linspace(self.limits_np[0][0], self.limits_np[1][0], steps=400)
        ys = torch.linspace(self.limits_np[0][1], self.limits_np[1][1], steps=400)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        sdf = self.compute_sdf(torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, self.dim), reshape_shape=X.shape)
        sdf = to_numpy(sdf)
        ctf = ax.contourf(X, Y, sdf)
        if fig is not None:
            fig.colorbar(ctf, orientation='vertical')
        ax.set_xlim(self.limits_np[0][0], self.limits_np[1][0])
        ax.set_ylim(self.limits_np[0][1], self.limits_np[1][1])
        if self.dim == 3:
            ax.set_zlim(self.limits_np[0][2], self.limits_np[1][2])
        ax.set_aspect('equal')

    def get_rrt_connect_params(self):
        raise NotImplementedError

    def get_sgpmp_params(self):
        raise NotImplementedError

    def get_gpmp_params(self):
        raise NotImplementedError
