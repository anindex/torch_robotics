import numpy as np
from matplotlib import pyplot as plt

from torch_robotics.environments import EnvTableShelf
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvTableShelfExtraObjects(EnvTableShelf):

    def __init__(self, tensor_args=None, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.5, 0.2, 0.25 / 2],
                    ]),
                np.array(
                    [
                        [0.25, 0.25, 0.25],
                    ]
                )
                ,
                tensor_args=tensor_args
            )
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, 'tableshelf-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvTableShelfExtraObjects(
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
