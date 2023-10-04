import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian

from torch_robotics.environments import EnvSimple2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSimple2DExtraObjects(EnvSimple2D):

    def __init__(self, tensor_args=None, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [-0.15, 0.15],
                        [-0.075, -0.85],
                        [-0.1, -0.1],
                        [0.45, -0.1],
                        [0.5, 0.35],
                        [-0.6, -0.85],
                        [0.05, 0.85],
                        [-0.8, 0.15],
                        [0.8, -0.8],
                    ]),
                np.array(
                    [
                        0.05,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                    ]
                )
                ,
                tensor_args=tensor_args
            ),
            MultiBoxField(
                np.array(
                    [
                        [0.45, -0.1],
                        [-0.25, -0.5],
                        [0.8, 0.1],
                    ]
                ),
                np.array(
                    [
                        [0.2, 0.2],
                        [0.15, 0.15],
                        [0.15, 0.15],
                    ]
                )
                ,
                tensor_args=tensor_args
            )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'dense2d-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvSimple2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
