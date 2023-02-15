from abc import ABC, abstractmethod
import torch


class Polytope(ABC):
    """
    Polytope represents workspace objects in N-D.
    """

    def __init__(self, tensor_args=None):
        if tensor_args is None:
            tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.tensor_args = tensor_args

    @abstractmethod
    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
        raise NotImplementedError()

    def zero_grad(self):
        pass


class Sphere(Polytope):
    
    def __init__(self, centers, radii, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Centers of the spheres.
            radii : numpy array
                Radii of the spheres.
        """
        super().__init__(tensor_args=tensor_args)
        self.centers = torch.tensor(centers, **self.tensor_args)
        self.radii = torch.tensor(radii, **self.tensor_args).squeeze()

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
        return torch.min(torch.norm(x.unsqueeze(-2) - self.centers.unsqueeze(0), dim=-1) - self.radii.unsqueeze(0), dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.radii.grad = None

    def __repr__(self):
        return f"MultiSphere(centers={self.centers}, radii={self.radii})"


class Ellipsoid(Polytope):

    def __init__(self, center, radii, tensor_args=None):
        """
        Axis aligned ellipsoid.
        Parameters
        ----------
            center : numpy array
                Center of the ellipsoid.
            radii : numpy array
                Radii of the ellipsoid.
        """
        super().__init__(tensor_args=tensor_args)
        self.center = torch.tensor(center, **self.tensor_args)
        self.radii = torch.tensor(radii, **self.tensor_args)

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
        return torch.norm((x - self.center) / self.radii, dim=-1) - 1

    def zero_grad(self):
        self.center.grad = None
        self.radii.grad = None

    def __repr__(self):
        return f"Ellipsoid(center={self.center}, radii={self.radii})"


class Box(Polytope):

    def __init__(self, center, size, tensor_args=None):
        """
        Parameters
        ----------
            center : numpy array
                Center of the box.
            size : numpy array
                Size of the box.
        """
        super().__init__(tensor_args=tensor_args)
        self.center = torch.tensor(center, **self.tensor_args)
        self.size = torch.tensor(size, **self.tensor_args)

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
        return torch.max(torch.abs(x - self.center) - self.size, dim=-1)[0]

    def zero_grad(self):
        self.center.grad = None
        self.size.grad = None

    def __repr__(self):
        return f"Box(center={self.center}, size={self.size})"


class Cylinder(Polytope):

    def __init__(self, center, radius, height, tensor_args=None):
        """
        Parameters
        ----------
            center : numpy array
                Center of the cylinder.
            radius : float
                Radius of the cylinder.
            height : float
                Height of the cylinder.
        """
        super().__init__(tensor_args=tensor_args)
        self.center = torch.tensor(center, **self.tensor_args)
        self.radius = torch.tensor(radius, **self.tensor_args)
        self.height = torch.tensor(height, **self.tensor_args)

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
        x = x - self.center
        x_proj = x[:, :2]
        x_proj_norm = torch.norm(x_proj, dim=-1)
        x_proj_norm = torch.where(x_proj_norm > 0, x_proj_norm, torch.ones_like(x_proj_norm))
        x_proj = x_proj / x_proj_norm[:, None]
        x_proj = x_proj * self.radius
        x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
        return torch.norm(x - x_proj, dim=-1) - self.radius

    def zero_grad(self):
        self.center.grad = None
        self.radius.grad = None
        self.height.grad = None

    def __repr__(self):
        return f"Cylinder(center={self.center}, radius={self.radius}, height={self.height})"


class Capsule(Polytope):

    def __init__(self, center, radius, height, tensor_args=None):
        """
        Parameters
        ----------
            center : numpy array
                Center of the capsule.
            radius : float
                Radius of the capsule.
            height : float
                Height of the capsule.
        """
        super().__init__(tensor_args=tensor_args)
        self.center = torch.tensor(center, **self.tensor_args)
        self.radius = torch.tensor(radius, **self.tensor_args)
        self.height = torch.tensor(height, **self.tensor_args)

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array
        """
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

