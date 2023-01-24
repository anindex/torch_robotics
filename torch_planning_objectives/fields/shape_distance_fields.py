from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_kinematics_tree.geometrics.utils import exp_map_so3, rot_mat_to_euler
from torch_kinematics_tree.geometrics.quaternion import q_to_rotation_matrix, q_to_euler


class Shape(ABC):
    """
    Shape represents workspace objects in 2D or 3D.
    The contour, distance and gradient and hessian of the distance
    function are represented as analytical or other type of functions.
    The implementations should return a set of points on the
    contour, to allow easy drawing.
    """

    def __init__(self, tensor_args=None):
        if tensor_args is None:
            tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.tensor_args = tensor_args

    @abstractmethod
    def compute_distance(self, x):
        """
        Returns the sign distance at x.
        Parameters
        ----------
            x : numpy array
        """
        raise NotImplementedError()


class MultiPoints(Shape):

    def __init__(self,
                 points=None,
                 tensor_args=None):
        super().__init__(tensor_args=tensor_args)
        if points.ndim == 1:
            points = points.unsqueeze(0)
        self.points = points

    def compute_distance(self, x):
        x = x.unsqueeze(-2)
        return torch.linalg.norm(x - self.points, dim=-1)
    
    def compute_cost(self, x, **kwargs):
        x = x.unsqueeze(-2)
        return torch.linalg.norm(x - self.points, dim=-1).sum(-1)  # sum over num goals


class LineSegment(Shape):
    """
    A segment defined with an origin, length and orientation, no batch
    """

    def __init__(self,
                 origin=None,
                 orientation=None,
                 length=1.,
                 p1=None,
                 p2=None,
                 tensor_args=None):
        super().__init__(tensor_args=tensor_args)
        if p1 is not None and p2 is not None:
            """ Initialize from end points """
            self.p1 = torch.tensor(p1, **self.tensor_args)
            self.p2 = torch.tensor(p2, **self.tensor_args)
            assert self.p1.shape[-1] == self.p2.shape[-1]
            self.dim = self.p1.shape[-1]
            self.origin = (self.p1 + self.p2) / 2.
            p12 = self.p1 - self.p2
            self.length = torch.linalg.norm(p12, dim=-1)
            if self.dim == 2:
                self.orientation = torch.arctan2(p12[1], p12[0])
            elif self.dim == 3:  # TODO
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            """  Inialize using orientation and length """
            self.origin = torch.tensor(origin, **self.tensor_args)
            self.dim = self.origin.shape[-1]
            if orientation.ndim == 1:
                self.orientation = torch.tensor(orientation, **self.tensor_args)
            else:  # TODO
                raise NotImplementedError()
            self.length = length
            self.p1, self.p2 = self.end_points()

    def end_points(self):
        if self.dim == 2:
            p0 = .5 * self.length * torch.tensor(
                [torch.cos(self.orientation), torch.sin(self.orientation)], **self.tensor_args)
        elif self.dim == 3:  # TODO
            raise NotImplementedError()
        p1 = self.origin + p0
        p2 = self.origin - p0
        return p1, p2

    def closest_point(self, x):
        """
        Compute the closest point by projecting to the infite line
        and then checking if the point lines on the segment.
        """
        assert x.shape[-1] == self.origin.shape[-1]
        u = self.p2 - self.p1
        v = x - self.p1
        d = (v @ u) / torch.dot(u, u)
        p = self.p1 + torch.outer(d, u)
        p[d <= 0.] = self.p1
        p[d >= 1.] = self.p2
        return p

    def compute_distance(self, x):
        p = self.closest_point(x)
        return torch.linalg.norm(p - x, dim=-1)


class MultiLineSegment(Shape):

    def __init__(self, segments=None, tensor_args=None, **obst_params):
        super().__init__(tensor_args=tensor_args)
        self.segments = segments
        if segments is not None:
            self.dim = vertices.shape[-1]
        self.obst_type = obst_params.get('obst_type', 'rbf')
        self.buffer = obst_params.get('buffer', 0.)
        self.segment_width = obst_params.get('segment_width', 25)
    
    def set_segments(self, vertices, widths=None):
        self.dim = vertices.shape[-1]
        self.segments = []
        for i in range(1, vertices.shape[0]):
            self.segments.append(LineSegment(p1=vertices[i-1], p2=vertices[i]))
        self.buff = torch.tensor(
                [self.buffer] * vertices.shape[0], **self.tensor_args
            )
        if widths is not None:
            self.widths = torch.tensor(widths, **self.tensor_args)
        else:
            self.widths = torch.tensor(
                [self.segment_width] * (vertices.shape[0] - 1), **self.tensor_args
            )

    def compute_distance(self, x):
        d = torch.zeros(x.shape[0], len(self.segments), **self.tensor_args)
        for i, segment in enumerate(self.segments):
            d[:, i] = segment.compute_distance(x)
        return d

    def compute_cost(self, X, **kwargs):
        batch_dim = X.shape[:-1]
        X = X.reshape((-1, self.dim))
        if self.segments is None:
            return torch.zeros(batch_dim, **self.tensor_args)
        if self.obst_type == 'sdf':
            costs = self.get_sdf(X)
        elif self.obst_type == 'rbf':
            costs = self.get_rbf(X)
        else:
            raise NotImplementedError
        return costs.reshape(batch_dim)

    def get_sdf(self, X):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        dists = self.compute_distance(X)
        sdf = (self.buff[None, :] - dists).sum(-1)
        return sdf

    def get_rbf(self, X):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        diff_sq = torch.square(self.compute_distance(X))
        rbf = torch.exp(-0.5 * diff_sq / torch.square(self.widths[None, :])).sum(-1)
        return rbf

    def plot_map(
            self,
            xy_lim=(-10, 10),
            res=100,
            save_path=None,
    ):
        import matplotlib.pyplot as plt
        ax = plt.gca()
        x = torch.linspace(xy_lim[0], xy_lim[1], res)
        y = torch.linspace(xy_lim[0], xy_lim[1], res)
        X, Y = torch.meshgrid(x, y)
        grid = torch.stack((X, Y), dim=-1)
        # Z = self.get_sdf(grid.reshape((-1, 2))).reshape((res, res))
        Z = self.get_rbf(grid.reshape((-1, 2))).reshape((res, res))

        # plt.imshow(Z.cpu().numpy(), cmap='viridis', origin='lower')
        cs = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), 10, cmap=plt.cm.binary, origin='lower')
        plt.colorbar()
        ax.xlim = xy_lim
        ax.ylim = xy_lim

        # if save_path is not None:
        #     plt.savefig(save_path) 


class MultiSphere(Shape):

    def __init__(
            self,
            tensor_args=None,
            **obst_params
    ):
        super().__init__(tensor_args=tensor_args)
        self.num_obst = obst_params.get('num_obst', 0)
        self.obst_type = obst_params.get('obst_type', 'rbf')
        self.buffer = obst_params.get('buffer', 0.)
        self.centers = None
        self.radii = None

    def set_obst(self, centers, radii):
        self.centers = torch.tensor(centers, **self.tensor_args)
        self.radii = torch.tensor(radii, **self.tensor_args)
        self.num_obst, self.dim = self.centers.shape[0], self.centers.shape[1]
        self.buff = torch.tensor(
                [self.buffer] * self.num_obst, **self.tensor_args
            )

    def compute_distance(self, X):
        return (torch.linalg.norm(X[:, None] - self.centers[None, :], dim=-1) - self.radii[None, :]).min(-1)

    def compute_cost(self, X, **kwargs):
        batch_dim = X.shape[:-1]
        X = X.reshape((-1, self.dim))
        if self.centers is None or self.radii is None:
            return torch.zeros(batch_dim, **self.tensor_args)
        if self.obst_type == 'sdf':
            costs = self.get_sdf(X, **kwargs)
        elif self.obst_type == 'rbf':
            costs = self.get_rbf(X, **kwargs)
        else:
            raise NotImplementedError
        return costs.reshape(batch_dim)

    def get_sdf(self, X):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        dists_to_centers = torch.linalg.norm(X[:, None] - self.centers[None, :], dim=-1)
        # sdf is the minimum over sdf of all circles https://jasmcole.com/2019/10/03/signed-distance-fields/
        sdf = dists_to_centers - (self.radii[None, :] + self.buff[None, :])
        sdf = sdf.min(-1)[0]
        return sdf

    def get_rbf(self, X, **kwargs):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        diff_sq = torch.square(X[:, None] - self.centers[None, :]).sum(-1)
        rbf = torch.exp(-0.5 * diff_sq / torch.square(self.radii[None, :]))
        return rbf.sum(-1)

    def plot_map(
            self,
            xy_lim=(-10, 10),
            res=100,
            save_path=None,
    ):
        import matplotlib.pyplot as plt
        ax = plt.gca()
        x = torch.linspace(xy_lim[0], xy_lim[1], res)
        y = torch.linspace(xy_lim[0], xy_lim[1], res)
        X, Y = torch.meshgrid(x, y)
        grid = torch.stack((X, Y), dim=-1)
        Z = self.get_sdf(grid.reshape((-1, 2))).reshape((res, res))
        # Z = self.get_rbf(grid.reshape((-1, 2))).reshape((res, res))

        # plt.imshow(Z.cpu().numpy(), cmap='viridis', origin='lower')
        cs = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(), 10, cmap=plt.cm.binary, origin='lower')
        plt.colorbar()
        ax.xlim = xy_lim
        ax.ylim = xy_lim

        # if save_path is not None:
        #     plt.savefig(save_path)


class Box3D(Shape):

    def __init__(self, box_centers, tensor_args=None, **obst_params):
        super().__init__(tensor_args=tensor_args)
        self.box_centers = box_centers
        self.obst_type = obst_params.get('obst_type', 'rbf')
        self.from_link = obst_params.get('from_link', 5)
        self.box_buffer = obst_params.get("box_buffer", 0.025)
        self.box_bounds = obst_params.get("box_bounds", torch.tensor(
            [
                [
                    [
                        [0.15, 0.15, 0.0],
                        [0.15, 0.15, 0.24],
                        [-0.15, 0.15, 0.0],
                        [-0.15, 0.15, 0.24],
                        [-0.15, -0.15, 0.0],
                        [-0.15, -0.15, 0.24],
                        [0.15, -0.15, 0.0],
                        [0.15, -0.15, 0.24],
                    ]
                ]
            ],
        dtype=torch.float32))
    
    def set_boxes(self, box_centers):
        self.box_centers = box_centers

    def compute_distance(self, x):
        closest_point2wall = x.unsqueeze(-2).repeat(1, 1, 5 * 4, 1)
        bounds = (self.box_centers.unsqueeze(-2) + self.box_bounds).unsqueeze(1)  # (batch, 1, num_boxes, 8, 3)
        closest_point2wall[..., ::5, :] = torch.clamp(
            closest_point2wall[..., ::5, :],
            min=bounds[..., 4, :],
            max=bounds[..., 0, :],
        )
        for idx, [i_min, i_max] in enumerate(zip([2, 4, 4, 6], [0, 2, 6, 0])):
            closest_point2wall[..., idx + 1 :: 5, :] = torch.clamp(
                closest_point2wall[..., idx + 1 :: 5, :],
                min=bounds[..., i_min, :],
                max=bounds[..., i_max + 1, :],
            )
        dist2walls = torch.linalg.norm(x.unsqueeze(2) - closest_point2wall, dim=-1)
        return dist2walls.min(-1)[0]

    def compute_cost(self, X, **kwargs):
        if self.box_centers is None:
            return np.zeros(X.shape[0])
        if self.obst_type == 'sdf':
            costs = self.get_sdf(X)
        elif self.obst_type == 'rbf':
            costs = self.get_rbf(X)
        else:
            raise NotImplementedError
        return costs

    def get_sdf(self, X):
        """
        Parameters
        ----------
        X : tensor

        Returns
        -------

        """
        raise NotImplementedError
        dists = self.compute_distance(X)
        sdf = (self.box_buffer - dists).sum(-1)
        return sdf

    def get_rbf(self, X):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        diff_sq = torch.square(self.compute_distance(X))
        rbf = torch.exp(-0.5 * diff_sq / (self.box_buffer ** 2))
        return rbf.sum(-1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vertices = np.array([
        [0, 0],
        [1280, 0],
        [1280, 320],
        [0, 320],
        [0, 0]
    ])
    shape = MultiLineSegment()
    shape.set_segments(vertices)
    shape.plot_map(xy_lim=(0, 1280))
    plt.show()

    centers = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    radii = np.array([
        0.1,
        0.2,
        0.3
    ])
    shape = MultiSphere()
    shape.set_obst(centers=centers, radii=radii)
    shape.plot_map(xy_lim=(-5, 5))
    plt.show()
