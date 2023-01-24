from abc import ABC, abstractmethod
import torch
import numpy as np
import copy
import yaml

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


class LinkSelfDistanceField(DistanceField):

    def __init__(self, margin=0.03, device='cpu'):
        self.margin = margin
        self.device = device

    def compute_distance(self, link_tensor):  # position tensor
        link_tensor = link_tensor[..., :3, -1]
        return torch.linalg.norm(link_tensor.unsqueeze(-2) - link_tensor.unsqueeze(-3), dim=-1).sum((-1, -2))

    def compute_cost(self, link_tensor, **kwargs):   # position tensor
        link_tensor = link_tensor[..., :3, -1]
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
