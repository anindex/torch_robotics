import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvPlanar2Link(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        circles = np.array([
            (0.2, 0.5, 0.3),
            (0.3, 0.15, 0.1),
            (0.5, 0.5, 0.1),
            (0.3, -0.5, 0.2),
            (-0.5, 0.5, 0.3),
            (-0.5, -0.5, 0.3),
        ])
        spheres = MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)
        obj_field = ObjectField([spheres], 'planar2link-spheres')
        obj_list = [obj_field]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environment limits
            obj_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvPlanar2Link(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = plt.subplots()
    xs = torch.linspace(-1, 1, steps=400)
    ys = torch.linspace(-1, 1, steps=400)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X_flat = torch.flatten(X)
    Y_flat = torch.flatten(Y)
    sdf = None
    for obj in env.obj_list:
        sdf_obj = obj.compute_signed_distance(torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, 2))
        sdf_obj = sdf_obj.reshape(X.shape)
        if sdf is None:
            sdf = sdf_obj
        else:
            sdf = torch.minimum(sdf, sdf_obj)
    ctf = ax.contourf(X, Y, sdf)
    fig.colorbar(ctf, orientation='vertical')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
