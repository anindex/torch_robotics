from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy


class PrimitiveShapeField(ABC):
    """
    Represents a primitive object in N-D.
    """

    def __init__(self, dim=3, tensor_args=None):
        self.dim = dim
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS
        self.tensor_args = tensor_args

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array (batch, dim) or (batch, horizon, dim)
        """
        return self.compute_signed_distance_impl(x)

    @abstractmethod
    def compute_signed_distance_impl(self, x):
        raise NotImplementedError()

    def zero_grad(self):
        pass

    def obstacle_collision_check(self, obst_map):
        valid = True
        obst_map_test = self.add_to_occupancy_map(deepcopy(obst_map))
        if np.any(obst_map_test.map > 1):
            valid = False
        return valid

    def point_collision_check(self, obst_map, pts):
        valid = True
        if pts is not None:
            obst_map_test = self.add_to_occupancy_map(np.copy(obst_map))
            for pt in pts:
                if obst_map_test[ceil(pt[0]), ceil(pt[1])] >= 1:
                    valid = False
                    break
        return valid

    @abstractmethod
    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to an occupancy grid
        raise NotImplementedError

    @abstractmethod
    def render(self, ax):
        raise NotImplementedError


class MultiSphereField(PrimitiveShapeField):

    def __init__(self, centers, radii, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Centers of the spheres.
            radii : numpy array
                Radii of the spheres.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.radii = to_torch(radii, **self.tensor_args)

    def __repr__(self):
        return f"Sphere(centers={self.centers}, radii={self.radii})"

    def compute_signed_distance_impl(self, x):
        distance_to_centers = torch.norm(x.unsqueeze(-2) - self.centers.unsqueeze(0), dim=-1)
        sdfs = distance_to_centers - self.radii.unsqueeze(0)
        return torch.min(sdfs, dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.radii.grad = None

    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to occupancy map
        for center, radius in zip(self.centers, self.radii):
            n_dim = len(center)
            # Convert dims to cell indices
            c_r = ceil(radius / obst_map.cell_size)
            c_x = ceil(center[0] / obst_map.cell_size)
            c_y = ceil(center[1] / obst_map.cell_size)

            # centers in cell indices
            obst_map_origin_xi = obst_map.origin[0]
            obst_map_origin_yi = obst_map.origin[1]

            c_x_cell = c_x + obst_map_origin_xi
            c_y_cell = c_y + obst_map_origin_yi

            obst_map_x_dim = obst_map.dims[0]
            obst_map_y_dim = obst_map.dims[1]

            if n_dim == 3:
                c_z = ceil(center[2] / obst_map.cell_size)
                obst_map_origin_zi = obst_map.origin[2]
                c_z_cell = c_z + obst_map_origin_zi
                obst_map_z_dim = obst_map.dims[2]

            for i in range(c_x_cell - 2 * c_r, c_x_cell + 2 * c_r):
                if i < 0 or i >= obst_map_x_dim:
                    continue
                for j in range(c_y_cell - 2 * c_r, c_y_cell + 2 * c_r):
                    if j < 0 or j >= obst_map_y_dim:
                        continue

                    if n_dim == 3:
                        for k in range(c_z_cell - 2 * c_r, c_z_cell + 2 * c_r):
                            if k < 0 or k >= obst_map_z_dim:
                                continue

                            p = torch.tensor([(i - obst_map_origin_xi) * obst_map.cell_size,
                                              (j - obst_map_origin_yi) * obst_map.cell_size,
                                              (k - obst_map_origin_zi) * obst_map.cell_size
                                              ],
                                             **self.tensor_args)

                            if self.is_inside(p, center, radius):
                                obst_map.map[j, i, k] += 1

                    else:
                        p = torch.tensor([(i - obst_map_origin_xi) * obst_map.cell_size,
                                          (j - obst_map_origin_yi) * obst_map.cell_size],
                                         **self.tensor_args)

                        if self.is_inside(p, center, radius):
                            obst_map.map[j, i] += 1
        return obst_map

    @staticmethod
    def is_inside(p, center, radius):
        # Check if point p is inside the discretized sphere
        return torch.linalg.norm(p - center) <= radius

    def render(self, ax):
        for center, radius in zip(self.centers, self.radii):
            center = to_numpy(center)
            radius = to_numpy(radius)
            if ax.name == '3d':
                u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
                x = radius * (np.cos(u) * np.sin(v))
                y = radius * (np.sin(u) * np.sin(v))
                z = radius * np.cos(v)
                ax.plot_surface(x + center[0], y + center[1], z + center[2], cmap='gray', alpha=1)
            else:
                circle = plt.Circle((center[0], center[1]), radius, color='gray', linewidth=0, alpha=1)
                ax.add_patch(circle)


class MultiBoxField(PrimitiveShapeField):

    def __init__(self, centers, sizes, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Center of the boxes.
            sizes: numpy array
                Sizes of the boxes.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.sizes = to_torch(sizes, **self.tensor_args)
        self.half_sizes = self.sizes / 2

    def __repr__(self):
        return f"Box(centers={self.centers}, sizes={self.sizes})"

    def compute_signed_distance_impl(self, x):
        distance_to_centers = torch.abs(x.unsqueeze(-2) - self.centers.unsqueeze(0))
        sdfs = torch.max(distance_to_centers - self.half_sizes.unsqueeze(0), dim=-1)[0]
        return torch.min(sdfs, dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.sizes.grad = None
        self.half_sizes.grad = None

    def add_to_occupancy_map(self, obst_map):
        for center, size in zip(self.centers, self.sizes):
            n_dim = len(center)
            width = size[0]
            height = size[1]
            # Convert dims to cell indices
            w = ceil(width / obst_map.cell_size)
            h = ceil(height / obst_map.cell_size)
            c_x = ceil(center[0] / obst_map.cell_size)
            c_y = ceil(center[1] / obst_map.cell_size)

            obst_map_origin_xi = obst_map.origin[0]
            obst_map_origin_yi = obst_map.origin[1]

            if n_dim == 2:
                obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi
                ] += 1
            else:
                depth = size[2]
                c_z = ceil(center[2] / obst_map.cell_size)
                obst_map_origin_zi = obst_map.origin[2]
                d = ceil(depth / obst_map.cell_size)
                obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi,
                c_z - ceil(d / 2.) + obst_map_origin_zi:c_z + ceil(d / 2.) + obst_map_origin_zi
                ] += 1
        return obst_map

    def render(self, ax):
        def get_cube():
            phi = np.arange(1, 10, 2) * np.pi / 4
            Phi, Theta = np.meshgrid(phi, phi)
            x = np.cos(Phi) * np.sin(Theta)
            y = np.sin(Phi) * np.sin(Theta)
            z = np.cos(Theta) / np.sqrt(2)
            return x, y, z

        if ax.name == '3d':
            x, y, z = get_cube()
            for center, size in zip(self.centers, self.sizes):
                cx, cy, cz = to_numpy(center)
                a, b, c = to_numpy(size)
                ax.plot_surface(cx + x * a, cy + y * b, cz + z * c, cmap='gray', alpha=0.25)
        else:
            for center, size in zip(self.centers, self.sizes):
                cx, cy = to_numpy(center)
                a, b = to_numpy(size)
                rectangle = plt.Rectangle((cx - a / 2, cy - b / 2), a, b, color='gray', linewidth=0, alpha=1)
                ax.add_patch(rectangle)


class MultiInfiniteCylinderField(PrimitiveShapeField):

    def __init__(self, centers, radii, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Centers of the cylinders.
            radii : numpy array
                Radii of the cylinders.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = torch.tensor(centers, **self.tensor_args)
        self.radii = torch.tensor(radii, **self.tensor_args)

    def __repr__(self):
        return f"InfiniteCylinder(centers={self.centers}, radii={self.radii})"

    def compute_signed_distance_impl(self, x):
        # treat it like a circle in 2d
        distance_to_centers_2d = torch.norm(x[..., :2].unsqueeze(-2) - self.centers.unsqueeze(0)[..., :2], dim=-1)
        sdfs = distance_to_centers_2d - self.radii.unsqueeze(0)
        return torch.min(sdfs, dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.radii.grad = None

    def add_to_occupancy_map(self, obst_map):
        raise NotImplementedError

    def render(self, ax):
        # https://stackoverflow.com/a/49311446
        def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
            z = np.linspace(0, height_z, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + center_x
            y_grid = radius * np.sin(theta_grid) + center_y
            return x_grid, y_grid, z_grid

        for center, radius in zip(self.centers, self.radii):
            cx, cy, cz = to_numpy(center)
            radius = to_numpy(radius)
            xc, yc, zc = data_for_cylinder_along_z(cx, cy, radius, 1.5)  # add height just for visualization
            ax.plot_surface(xc, yc, zc, cmap='gray', alpha=0.75)


class MultiCylinderField(MultiInfiniteCylinderField):

    def __init__(self, centers, radii, heights, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Centers of the cylinders.
            radii : numpy array
                Radii of the cylinders.
            heights : numpy array
                Heights of the cylinders.
        """
        super().__init__(centers, radii, tensor_args=tensor_args)
        self.heights = torch.tensor(heights, **self.tensor_args)
        self.half_heights = self.heights / 2

    def __repr__(self):
        return f"Cylinder(centers={self.centers}, radii={self.radii}, heights={self.heights})"

    def compute_signed_distance_impl(self, x):
        raise NotImplementedError
        x = x - self.center
        x_proj = x[:, :2]
        x_proj_norm = torch.norm(x_proj, dim=-1)
        x_proj_norm = torch.where(x_proj_norm > 0, x_proj_norm, torch.ones_like(x_proj_norm))
        x_proj = x_proj / x_proj_norm[:, None]
        x_proj = x_proj * self.radius
        x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
        return torch.norm(x - x_proj, dim=-1) - self.radius

    def zero_grad(self):
        super().zero_grad()
        self.heights.grad = None

    def add_to_occupancy_map(self, obst_map):
        raise NotImplementedError


class MultiEllipsoidField(PrimitiveShapeField):

    def __init__(self, centers, radii, tensor_args=None):
        """
        Axis aligned ellipsoid.
        Parameters
        ----------
            center : numpy array
                Center of the ellipsoid.
            radii : numpy array
                Radii of the ellipsoid.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.center = torch.tensor(centers, **self.tensor_args)
        self.radii = torch.tensor(radii, **self.tensor_args)

    def compute_signed_distance_impl(self, x):
        return torch.norm((x - self.center) / self.radii, dim=-1) - 1

    def zero_grad(self):
        self.center.grad = None
        self.radii.grad = None

    def __repr__(self):
        return f"Ellipsoid(center={self.center}, radii={self.radii})"


class MultiCapsuleField(PrimitiveShapeField):

    def __init__(self, centers, radii, heights, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Center of the capsule.
            radiii : float
                Radius of the capsule.
            heights : float
                Height of the capsule.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.center = torch.tensor(centers, **self.tensor_args)
        self.radius = torch.tensor(radii, **self.tensor_args)
        self.height = torch.tensor(heights, **self.tensor_args)

    def compute_signed_distance_impl(self, x):
        x = x - self.center
        x_proj = x[:, :2]
        x_proj_norm = torch.norm(x_proj, dim=-1)
        x_proj_norm = torch.where(x_proj_norm > 0, x_proj_norm, torch.ones_like(x_proj_norm))
        x_proj = x_proj / x_proj_norm[:, None]
        x_proj = x_proj * self.radius
        x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
        x_proj = torch.norm(x - x_proj, dim=-1) - self.radius
        x_proj = torch.where(x_proj > 0, x_proj, torch.zeros_like(x_proj))
        x_proj = torch.where(x_proj < self.height, x_proj, torch.ones_like(x_proj) * self.height)
        x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
        return torch.norm(x - x_proj, dim=-1) - self.radius

    def zero_grad(self):
        self.center.grad = None
        self.radius.grad = None
        self.height.grad = None

    def __repr__(self):
        return f"Capsule(center={self.center}, radius={self.radius}, height={self.height})"


class MeshField(PrimitiveShapeField):
    """
    Represents a mesh as a primitive shape.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


########################################################################################################################
class ObjectField(PrimitiveShapeField):

    def __init__(self, primitive_fields, name='object', pos=None, ori=None):
        """
        Holds an object made of primitives and manages its position and orientation in the environment.
        """
        self.name = name

        assert primitive_fields is not None
        super().__init__(dim=primitive_fields[0].dim, tensor_args=primitive_fields[0].tensor_args)
        self.fields = primitive_fields

        # position and orientation
        assert (pos is None and ori is None) or (pos.ndims == 2 and ori.ndims == 2)
        self.pos = torch.zeros((1, self.dim), **self.tensor_args) if pos is None else pos
        self.ori = torch.tensor([1, 0, 0, 0], **self.tensor_args).view((1, -1)) if ori is None else ori  # quat - wxyz

    def __repr__(self):
        return f"Scene(fields={self.fields})"

    def join_primitives(self):
        raise NotImplementedError

    def compute_signed_distance_impl(self, x):
        sdf_fields = []
        for field in self.fields:
            sdf_fields.append(field.compute_signed_distance_impl(x))
        return torch.min(torch.stack(sdf_fields, dim=-1), dim=-1)[0]

    def render(self, ax):
        for field in self.fields:
            field.render(ax)

    def add_to_occupancy_map(self, occ_map):
        for field in self.fields:
            field.add_to_occupancy_map(occ_map)

    def zero_grad(self):
        for field in self.fields:
            field.zero_grad()


if __name__ == '__main__':
    tensor_args = DEFAULT_TENSOR_ARGS
    spheres = MultiSphereField(torch.zeros(2, **tensor_args).view(1, -1),
                               torch.ones(1, **tensor_args).view(1, -1) * 0.3,
                               tensor_args=tensor_args)

    boxes = MultiBoxField(torch.zeros(2, **tensor_args).view(1, -1) + 0.5,
                          torch.ones(2, **tensor_args).view(1, -1) * 0.3,
                          tensor_args=tensor_args)

    obj_field = ObjectField([spheres, boxes])
    # obj_field = ObjectField([spheres])

    # Render objects
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    obj_field.render(ax)
    plt.show()

    # Render sdf
    fig, ax = plt.subplots()
    xs = torch.linspace(-1, 1, steps=200)
    ys = torch.linspace(-1, 1, steps=200)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X_flat = torch.flatten(X)
    Y_flat = torch.flatten(Y)
    sdf = obj_field.compute_signed_distance(torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, 2))
    sdf = sdf.reshape(X.shape)
    ctf = ax.contourf(X, Y, sdf)
    fig.colorbar(ctf, orientation='vertical')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
