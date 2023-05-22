import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiSphereField
from torch_robotics.environment.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class GridCircles2D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        obj_list = create_grid_spheres(rows=7, cols=7, heights=0, radius=0.1, tensor_args=tensor_args)

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = GridCircles2D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()
