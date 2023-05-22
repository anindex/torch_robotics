import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiSphereField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        spheres = MultiSphereField(torch.tensor([
                    [0.6, 0.3, 0.],
                    [0.5, 0.3, 0.5],
                    [-0.5, 0.25, 0.6],
                    [-0.6, -0.2, 0.4],
                    [-0.7, 0.1, 0.0],
                    [0.5, -0.45, 0.2],
                    [0.6, -0.35, 0.6],
                    [0.3, 0.0, 1.0],
                    ]),
                torch.tensor([
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                ]),
                tensor_args=tensor_args)

        obj_field = ObjectField([spheres], 'spheres')
        obj_list = [obj_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvSpheres3D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()
