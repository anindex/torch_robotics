import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3DExtraObjects(EnvSpheres3D):

    def __init__(self, tensor_args=None, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [0., 0.5, 0.5],
                        [0., -0.5, 0.],
                        [-0.25, -0.5, 0.5],
                        [-0.25, 0., 0.75],
                    ]),
                np.array(
                    [
                        0.15,
                        0.15,
                        0.15,
                        0.15,
                    ]
                )
                ,
                tensor_args=tensor_args
            ),
            # MultiBoxField(
            #     np.array(
            #         [
            #             [0.1, 0.35, 0.],
            #             [0.25, 0.0, 0.35],
            #         ]),
            #     np.array(
            #         [
            #             [0.25, 0.25, 0.25],
            #             [0.125, 0.125, 0.125],
            #         ]
            #     )
            #     ,
            #     tensor_args=tensor_args
            # )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'extra-objects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvSpheres3DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
