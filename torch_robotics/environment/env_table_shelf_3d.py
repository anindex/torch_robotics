from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


def create_table_object_field(tensor_args=None):
    centers = [(0., 0., 0.)]
    sizes = [(0.56, 0.90, 0.80)]
    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'table')


def create_shelf_field(tensor_args=None):
    width = 0.80
    height = 2.05
    depth = 0.28
    side_panel_width = 0.02

    shelf_width = width - 2*side_panel_width
    shelf_height = 0.015
    shelf_depth = depth

    # left panel
    centers = [(side_panel_width/2, depth/2, height/2)]
    sizes = [(side_panel_width, depth, height)]
    # right panel
    centers.append((side_panel_width + shelf_width + side_panel_width/2, depth/2, height/2))
    sizes.append((side_panel_width, depth, height))
    # back panel
    centers.append((side_panel_width + shelf_width/2, depth + side_panel_width/2, height/2))
    sizes.append((shelf_width, side_panel_width, height))

    # bottom shelf
    centers.append((side_panel_width + shelf_width/2, depth/2, shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # top shelf
    centers.append((side_panel_width + shelf_width/2, depth/2, height - shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # shelf 1
    first_shelf_height = 0.82
    centers.append((side_panel_width + shelf_width/2, depth/2, first_shelf_height + shelf_height/2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # next shelves
    plus_height_l = [0.23, 0.255, 0.225, 0.225]
    for plus_height in plus_height_l:
        center = list(copy(centers[-1]))
        center[-1] += plus_height
        centers.append(center)
        sizes.append((shelf_width, shelf_depth, shelf_height))

    centers = np.array(centers)
    main_center = np.array((side_panel_width + shelf_width/2, depth/2, height/2))
    centers -= main_center
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], 'shelf')


class EnvTableShelf3D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        # creates a table and shelf fields
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        shelf_obj_field = create_shelf_field(tensor_args=tensor_args)

        obj_list = [table_obj_field, shelf_obj_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvTableShelf3D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
