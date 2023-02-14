from abc import ABC, abstractmethod
import torch
import numpy as np
import copy
import yaml
import trimesh

from torch_kinematics_tree.geometrics.utils import transform_point, SE3_distance
from torch_kinematics_tree.utils.files import get_configs_path
from torch_planning_objectives.fields.utils.geom_types import tensor_sphere
from torch_planning_objectives.fields.utils.distance import find_link_distance, find_obstacle_distance


class DistanceField(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def compute_distance(self):
        pass

    @abstractmethod
    def compute_cost(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass 


class SphereDistanceField(DistanceField):
    """ This class holds a batched collision model where the robot is represented as spheres.
        All points are stored in the world reference frame, obtained by using update_pose calls.
    """

    def __init__(self, robot_collision_params, batch_size=1, device='cpu'):
        """ Initialize with robot collision parameters, look at franka_reacher.py for an example.
        Args:
            robot_collision_params (Dict): collision model parameters
            batch_size (int, optional): Batch size of parallel sdf computation. Defaults to 1.
            tensor_args (dict, optional): compute device and data type. Defaults to {'device':"cpu", 'dtype':torch.float32}.
        """        
        self.batch_dim = [batch_size, ]
        self.device = device

        self._link_spheres = None
        self._batch_link_spheres = None
        self.w_batch_link_spheres = None

        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
        self.margin = robot_collision_params.get('margin', 0.1)

        self.self_dist = None
        self.obst_dist = None

    def load_robot_collision_model(self, robot_collision_params):
        """Load robot collision model, called from constructor
        Args:
            robot_collision_params (Dict): loaded from yml file
        """        
        self.robot_links = robot_collision_params['link_objs']

        # load collision file:
        coll_yml = (get_configs_path() / robot_collision_params['collision_spheres']).as_posix()
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        self._link_spheres = []
        # we store as [n_link, n_dim]

        for j_idx, j in enumerate(self.robot_links):
            n_spheres = len(coll_params[j])
            link_spheres = torch.zeros((n_spheres, 4), device=self.device)
            for i in range(n_spheres):
                link_spheres[i, :] = tensor_sphere(coll_params[j][i][:3], coll_params[j][i][3], device=self.device, tensor=link_spheres[i])
            self._link_spheres.append(link_spheres)

    def build_batch_features(self, clone_objs=False, batch_dim=None):
        """clones poses/object instances for computing across batch. Use this once per batch size change to avoid re-initialization over repeated calls.
        Args:
            clone_objs (bool, optional): clones objects. Defaults to False.
            batch_size ([type], optional): batch_size to clone. Defaults to None.
        """

        if(batch_dim is not None):
            self.batch_dim = batch_dim
        if(clone_objs):
            self._batch_link_spheres = []
            for i in range(len(self._link_spheres)):
                _batch_link_i = self._link_spheres[i].view(tuple([1] * len(self.batch_dim) + list(self._link_spheres[i].shape)))
                _batch_link_i = _batch_link_i.repeat(tuple(self.batch_dim + [1, 1]))
                self._batch_link_spheres.append(_batch_link_i)
        self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres)

    def update_batch_robot_collision_objs(self, links_dict):
        '''update pose of link spheres
        Args:
        links_pos: bxnx3
        links_rot: bxnx3x3
        '''

        for i in range(len(self.robot_links)):
            link_H = links_dict[self.robot_links[i]].get_transform_matrix()
            link_pos, link_rot = link_H[..., :-1, -1], link_H[..., :3, :3]
            self.w_batch_link_spheres[i][..., :3] = transform_point(self._batch_link_spheres[i][..., :3], link_rot, link_pos.unsqueeze(-2))

    def check_collisions(self, obstacle_spheres=None):
        """Analytic method to compute signed distance between links.
        Args:
            link_trans ([tensor]): link translation as batch [b, 3]
            link_rot ([type]): link rotation as batch [b, 3, 3]
        Returns:
            [tensor]: signed distance [b, 1]
        """        
        n_links = len(self.w_batch_link_spheres)
        if self.self_dist is None:
            self.self_dist = torch.zeros(self.batch_dim + [n_links, n_links], device=self.device) - 100.0
        if self.obst_dist is None:
            self.obst_dist = torch.zeros(self.batch_dim + [n_links,], device=self.device) - 100.0
        dist = self.self_dist
        dist = find_link_distance(self.w_batch_link_spheres, dist)
        total_dist = dist.max(1)[0]
        if obstacle_spheres is not None:
            obst_dist = self.obst_dist
            obst_dist = find_obstacle_distance(obstacle_spheres, self.w_batch_link_spheres, obst_dist)
            total_dist += obst_dist
        return dist

    def compute_distance(self, links_dict, obstacle_spheres=None):
        self.update_batch_robot_collision_objs(links_dict)
        return self.check_collisions(obstacle_spheres).max(1)[0]
    
    def compute_cost(self, links_dict, obstacle_spheres=None):
        signed_dist = self.compute_distance(links_dict, obstacle_spheres=obstacle_spheres)
        return torch.exp(signed_dist / self.margin)

    def get_batch_robot_link_spheres(self):
        return self.w_batch_link_spheres

    def zero_grad(self):
        self.self_dist.detach_()
        self.self_dist.grad = None
        self.obst_dist.detach_()
        self.obst_dist.grad = None
        for i in range(len(self.robot_links)):
            self.w_batch_link_spheres[i].detach_()
            self.w_batch_link_spheres[i].grad = None
            self._batch_link_spheres[i].detach_()
            self._batch_link_spheres[i].grad = None


class LinkDistanceField(DistanceField):

    def __init__(self, field_type='rbf', clamp_sdf=False, num_interpolate=0, link_interpolate_range=[5, 7], device='cpu'):
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range
        self.device = device

    def compute_distance(self, link_tensor, obstacle_spheres=None, **kwargs):
        if obstacle_spheres is None:
            return 1e10
        link_tensor = link_tensor[..., :3, -1].unsqueeze(-2)
        obstacle_spheres = obstacle_spheres.unsqueeze(0)
        return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) - obstacle_spheres[..., 3]).sum((-1, -2)) 

    def compute_cost(self, link_tensor, obstacle_spheres=None, **kwargs):
        if obstacle_spheres is None:
            return 0
        link_tensor = link_tensor[..., :3, -1]
        link_dim = link_tensor.shape[:-1]
        if self.num_interpolate > 0:
            alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
            alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
            for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
                X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
                eval_sphere = X1 + (X2 - X1) * alpha
                link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
        link_tensor = link_tensor.unsqueeze(-2)
        obstacle_spheres = obstacle_spheres.unsqueeze(0)
        # signed distance field
        if self.field_type == 'rbf':
            return torch.exp(-0.5 * torch.square(link_tensor - obstacle_spheres[..., :3]).sum(-1) / torch.square(obstacle_spheres[..., 3])).sum((-1, -2))
        elif self.field_type == 'sdf':
            sdf = -torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) + obstacle_spheres[..., 3]
            if self.clamp_sdf:
                sdf = sdf.clamp(max=0.)
            return sdf.max(-1)[0].max(-1)[0]
        elif self.field_type == 'occupancy':
            return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) < obstacle_spheres[..., 3]).sum((-1, -2))

    def zero_grad(self):
        pass


class EmbodimentDistanceField(DistanceField):

    def __init__(self, self_margin=0.01, obst_margin=0.03, field_type='rbf', num_interpolate=0, link_interpolate_range=[2, 7], **kwargs):
        super().__init__(**kwargs)
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range
        self.self_margin = self_margin
        self.obst_margin = obst_margin
        self.field_type = field_type

    def self_distances(self, link_pos, **kwargs):  # position tensor
        dist_mat = torch.linalg.norm(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3), dim=-1)  # batch_dim x links x links
        # select lower triangular
        lower_indices = torch.tril_indices(dist_mat.shape[-1], dist_mat.shape[-1], offset=-1).unbind()
        distances = dist_mat[:, lower_indices[0], lower_indices[1]]  # batch_dim x (links * (links - 1) / 2)
        return distances

    def compute_self_cost(self, link_pos, field_type=None, margin=None, **kwargs):  # position tensor
        if field_type is None:
            field_type = self.field_type
        if margin is None:
            margin = self.self_margin
        if field_type == 'rbf':
            return torch.exp(torch.square(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3)).sum(-1) / (-margin**2 * 2)).sum((-1, -2))
        elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
            return -(self.self_distances(link_pos, **kwargs) - margin).min(-1)[0]
        elif field_type == 'occupancy':
            distances = self.self_distances(link_pos, **kwargs)  # batch_dim x links x links
            return (distances < margin).sum(-1)
        else:
            raise NotImplementedError('field_type {} not implemented'.format(field_type))

    def compute_self_collision(self, link_pos, margin=None, **kwargs):  # position tensor
        if margin is None:
            margin = self.self_margin
        distances = self.self_distances(link_pos, **kwargs)  # batch_dim x links x links
        any_self_collision = torch.any(distances < margin, dim=-1)
        return any_self_collision

    def obstacle_distances(self, link_pos, df_list=None, **kwargs):
        if df_list is None:
            return 1e10   # or infinity
        link_dim = link_pos.shape[:-1]
        link_pos = link_pos.reshape(-1, link_pos.shape[-1])  # flatten batch_dim and links
        dfs = [df.compute_distance(link_pos).view(link_dim) for df in df_list]  # df() returns batch_dim x links
        return torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links

    def compute_obstacle_collision(self, link_pos, margin=None, **kwargs):  # position tensor
        if margin is None:
            margin = self.obst_margin
        distances = self.obstacle_distances(link_pos, **kwargs).min(-1)[0].min(-1)[0]  # batch_dim
        any_collision = distances < margin
        return any_collision
    
    def compute_obstacle_cost(self, link_pos, df_list=None, field_type=None, margin=None, **kwargs):
        if df_list is None:
            return 0
        if field_type is None:
            field_type = self.field_type
        if margin is None:
            margin = self.obst_margin
        if field_type == 'rbf':
            return torch.exp(torch.square(self.obstacle_distances(link_pos, df_list, **kwargs)) / (-margin**2 * 2)).sum((-1, -2))
        elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
            return -(self.obstacle_distances(link_pos, df_list, **kwargs) - margin).min(-1)[0].min(-1)[0]
        elif field_type == 'occupancy':
            return (self.obstacle_distances(link_pos, df_list, **kwargs) < margin).sum((-1, -2))
        else:
            raise NotImplementedError('field_type {} not implemented'.format(field_type))

    def interpolate_links(self, link_pos):
        link_dim = link_pos.shape[:-1]
        alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_pos)[1:self.num_interpolate + 1]
        alpha = alpha.view(tuple([1] * len(link_dim) + [-1, 1]))  # 1 x 1 x 1 x ... x num_interpolate x 1
        X = link_pos[..., self.link_interpolate_range[0]:self.link_interpolate_range[1] + 1, :].unsqueeze(-2)  # batch_dim x num_interp_link x 1 x 3
        X_diff = torch.diff(X, dim=-3)  # batch_dim x (num_interp_link - 1) x 1 x 3
        X_interp = X[..., :-1, :, :] + X_diff * alpha  # batch_dim x (num_interp_link - 1) x num_interpolate x 3
        return torch.cat([link_pos, X_interp.flatten(-3, -2)], dim=-2)  # batch_dim x (num_link + (num_interp_link - 1) * num_interpolate) x 3

    def compute_distance(self, link_pos, df_list=None, **kwargs):  # position tensor
        if self.num_interpolate > 0:
            link_pos = self.interpolate_links(link_pos)
        
        self_distances = self.self_distances(link_pos, **kwargs).min(-1)[0]  # batch_dim
        obstacle_distances = self.obstacle_distances(link_pos, df_list, **kwargs).min(-1)[0].min(-1)[0]  # batch_dim
        return self_distances, obstacle_distances

    def compute_cost(self, link_pos, df_list=None, **kwargs):  # position tensor # batch_dim x num_links x 3
        if self.num_interpolate > 0:
            link_pos = self.interpolate_links(link_pos)

        self_cost = self.compute_self_cost(link_pos, **kwargs)
        obstacle_cost = self.compute_obstacle_cost(link_pos, df_list, **kwargs)
        return self_cost, obstacle_cost

    def zero_grad(self):
        pass


class LinkSelfDistanceField(DistanceField):

    def __init__(self, margin=0.03, num_interpolate=0, link_interpolate_range=[5, 7], device='cpu'):
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range
        self.margin = margin
        self.device = device

    def compute_distance(self, link_tensor):  # position tensor
        link_tensor = link_tensor[..., :3, -1]
        return torch.linalg.norm(link_tensor.unsqueeze(-2) - link_tensor.unsqueeze(-3), dim=-1).sum((-1, -2))

    def compute_cost(self, link_tensor, **kwargs):   # position tensor
        link_tensor = link_tensor[..., :3, -1]
        link_dim = link_tensor.shape[:-1]
        if self.num_interpolate > 0:
            alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
            alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
            for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
                X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
                eval_sphere = X1 + (X2 - X1) * alpha
                link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
        return torch.exp(torch.square(link_tensor.unsqueeze(-2) - link_tensor.unsqueeze(-3)).sum(-1) / (-self.margin**2 * 2)).sum((-1, -2))

    def zero_grad(self):
        pass


class FloorDistanceField(DistanceField):

    def __init__(self, margin=0.05, device='cpu'):
        self.margin = margin
        self.device = device

    def compute_distance(self, link_tensor):  # position tensor
        return link_tensor[..., 2, -1].mean(-1)   # z axis

    def compute_cost(self, link_tensor, **kwargs):  # position tensor
        return torch.exp(-0.5 * torch.square(link_tensor[..., 2, -1].mean(-1)) / self.margin**2)

    def zero_grad(self):
        pass


class EESE3DistanceField(DistanceField):

    def __init__(self, target_H, w_pos=1., w_rot=1., square=True, device='cpu'):
        self.target_H = target_H
        self.square = square
        self.w_pos = w_pos
        self.w_rot = w_rot
        self.device = device
    
    def update_target(self, target_H):
        self.target_H = target_H

    def compute_distance(self, link_tensor):  # position tensor
        return SE3_distance(link_tensor[..., -1, :, :], self.target_H, w_pos=self.w_pos, w_rot=self.w_rot)   # get EE as last link

    def compute_cost(self, link_tensor, **kwargs):  # position tensor
        dist = self.compute_distance(link_tensor).squeeze()
        if self.square:
            dist = torch.square(dist)
        return dist

    def zero_grad(self):
        pass


class SkeletonSE3DistanceField(DistanceField):

    def __init__(self, target_H=None, w_pos=1., w_rot=1., link_list=None, square=True, device='cpu'):
        self.target_H = target_H
        self.w_pos = w_pos
        self.w_rot = w_rot
        self.square = square
        self.link_list = link_list
        self.link_weights = None
        self.link_weights_ts = None
        self.device = device

    def set_link_weights(self, weight_dict):
        assert all(link in self.link_list for link in weight_dict)
        self.link_weights.update(weight_dict)
        total = sum(self.link_weights.values())
        weights = np.array([self.link_weights[l] / total for l in self.link_weights])
        self.link_weights_ts = torch.from_numpy(weights).to(self.device)

    def construct_target_from_configuration(self, model, q, link_list=None):
        self.device = model._device
        if isinstance(q, np.ndarray) or isinstance(q, list):
            q = torch.tensor(q, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        self.target_H = model.compute_forward_kinematics_link_list(q, return_dict=False, link_list=link_list)
        if link_list is None:
            if model.link_list is None:
                self.link_list = model.get_link_names()
            else:
                self.link_list = model.link_list
        else:
            self.link_list = link_list
        self.link_weights = {}
        for l in self.link_list:
            self.link_weights[l] = 1.
        self.link_weights_ts = torch.ones(len(self.link_list), device=self.device) / len(self.link_list)

    def compute_distance(self, link_tensor, **kwargs):  # assume link_tensor has index ordering like link_list
        link_list = kwargs.get('link_list', None)
        link_weights_ts = self.link_weights_ts
        target_H = self.target_H
        if link_list is not None:
            link_indices = [self.link_list.index(l) for l in link_list]
            link_tensor = link_tensor[..., link_indices, :, :]
            target_H = target_H[..., link_indices, :, :]
            link_weights_ts = link_weights_ts[link_indices] * len(self.link_list) / len(link_indices)  # reweight
        return (SE3_distance(link_tensor, target_H, w_pos=self.w_pos, w_rot=self.w_rot) * link_weights_ts).sum(-1)

    def compute_cost(self, link_tensor, **kwargs):  # position tensor
        dist = self.compute_distance(link_tensor, **kwargs).squeeze()
        if self.square:
            dist = torch.square(dist)
        return dist

    def zero_grad(self):
        pass



class MeshDistanceField(DistanceField):

    def __init__(self, mesh_file, margin=0.03, field_type='rbf', num_interpolate=0, link_interpolate_range=[], base_position=None, base_orientation=None, tensor_args=None) -> None:
        if tensor_args is None:
            tensor_args = dict(device='cpu', dtype=torch.float32)
        self.tensor_args = tensor_args
        self.field_type = field_type
        self.num_interpolate = num_interpolate
        self.link_interpolate_range = link_interpolate_range
        self.margin = margin
        if base_position is None:
            base_position = torch.zeros(3, **self.tensor_args)
        else:
            base_position = torch.tensor(base_position, **self.tensor_args)
        if base_orientation is None:
            base_orientation = torch.eye(3, **self.tensor_args)
        else:
            base_orientation = torch.tensor(base_orientation, **self.tensor_args)

        # Load mesh
        self.mesh = trimesh.load_mesh(mesh_file)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.dump()[0]
        self.vertices = torch.tensor(self.mesh.vertices, **self.tensor_args)
        self.faces = torch.tensor(self.mesh.faces, **self.tensor_args)
        # Transform mesh
        self.vertices = self.vertices @ base_orientation.T + base_position

    def compute_distance(self, points, sample=False, num_samples=100):
        if sample:
            indices = np.random.choice(self.vertices.shape[0], num_samples, replace=False)
            vertices = self.vertices[indices]
        else:
            vertices = self.vertices
        distances = torch.min(torch.norm(points.unsqueeze(-2) - vertices, dim=-1), dim=-1)[0]
        return distances

    def compute_cost(self, link_tensor, sample=False, num_samples=100, **kwargs):
        if sample:
            indices = np.random.choice(self.vertices.shape[0], num_samples, replace=False)
            vertices = self.vertices[indices]
        else:
            vertices = self.vertices
        link_tensor = link_tensor[..., :3, -1]
        link_dim = link_tensor.shape[:-1]
        if self.num_interpolate > 0:
            alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
            alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
            for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
                X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
                eval_sphere = X1 + (X2 - X1) * alpha
                link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
        link_tensor = link_tensor.unsqueeze(-2)
        vertices = vertices.unsqueeze(0)
        # signed distance field
        if self.field_type == 'rbf':
            return torch.exp(-0.5 * torch.square(link_tensor - vertices[..., :3]).sum(-1) / (self.margin ** 2)).sum((-1, -2))
        elif self.field_type == 'occupancy':
            return (torch.linalg.norm(link_tensor - vertices[..., :3], dim=-1) < self.margin[..., 3]).sum((-1, -2))

    def zero_grad(self):
        pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    mesh_file = 'models/chair.obj'
    mesh = MeshDistanceField(mesh_file)
    bounds = np.array(mesh.mesh.bounds)
    print(np.linalg.norm(bounds[1] - bounds[0]))
    print(mesh.mesh.centroid)
    points = torch.rand(100, 3)
    link_tensor = torch.rand(100, 10, 4, 4)
    start = time.time()
    distances = mesh.compute_distance(points)
    costs = mesh.compute_cost(link_tensor)
    print(time.time() - start)
