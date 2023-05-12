import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


def create_3d_rectangles_objects(tensor_args=None):
    rectangles_bl_tr = list()
    # bottom left xy, top right xy coordinates
    halfside = 0.2
    rectangles_bl_tr.append((-0.75, -1.0, -0.75 + halfside, -0.3))
    rectangles_bl_tr.append((-0.75, 0.3, -0.75 + halfside, 1.0))
    rectangles_bl_tr.append((-0.75, -0.15, -0.75 + halfside, 0.15))

    rectangles_bl_tr.append((-halfside/2 - 0.2, -1, halfside/2 - 0.2, -1+0.05))
    rectangles_bl_tr.append((-halfside/2 - 0.2, -1+0.2, halfside/2 - 0.2, 1-0.2))
    rectangles_bl_tr.append((-halfside/2 - 0.2, 1-0.05, halfside/2 - 0.2, 1))

    rectangles_bl_tr.append((-halfside / 2 + 0.2, -1.0, halfside / 2 + 0.2, -0.3))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, 0.3, halfside / 2 + 0.2, 1.0))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, -0.15, halfside / 2 + 0.2, 0.15))

    rectangles_bl_tr.append((0.75-halfside, -1, 0.75, -0.6))
    rectangles_bl_tr.append((0.75-halfside, -0.5, 0.75, -0.2))
    rectangles_bl_tr.append((0.75-halfside, -0.1, 0.75, 0.1))
    rectangles_bl_tr.append((0.75-halfside, 0.2, 0.75, 0.5))
    rectangles_bl_tr.append((0.75-halfside, 0.6, 0.75, 1))

    centers = []
    sizes = []
    for rectangle in rectangles_bl_tr:
        bl_x, bl_y, tr_x, tr_y = rectangle
        x = bl_x + abs(tr_x - bl_x) / 2
        y = bl_y + abs(tr_y - bl_y) / 2
        z = 0
        w = abs(tr_x - bl_x)
        h = abs(tr_y - bl_y)
        d = 1.95
        centers.append((x, y, z))
        sizes.append((w, h, d))

    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    obj_field = ObjectField([boxes], 'boxes')
    obj_list = [obj_field]
    return obj_list


class EnvMazeBoxes3D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        obj_list = create_3d_rectangles_objects(tensor_args=tensor_args)

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvMazeBoxes3D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()
