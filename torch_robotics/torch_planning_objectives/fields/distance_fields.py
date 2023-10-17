from abc import ABC, abstractmethod

import einops
import torch
from matplotlib import pyplot as plt

from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet
from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import torch.nn.functional as Functional


class DistanceField(ABC):
    def __init__(self, tensor_args=None):
        self.tensor_args = tensor_args

    def distances(self):
        pass

    def compute_collision(self):
        pass

    @abstractmethod
    def compute_distance(self, *args, **kwargs):
        pass

    def compute_cost(self, q, link_pos, *args, **kwargs):
        q_orig_shape = q.shape
        link_orig_shape = link_pos.shape
        if len(link_orig_shape) == 2:
            h = 1
            b, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b d -> b 1 d")  # add dimension of task space link
        elif len(link_orig_shape) == 3:
            h = 1
            b, t, d = link_orig_shape
        elif len(link_orig_shape) == 4:  # batch, horizon, num_links, 3  # position tensor
            b, h, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d -> (b h) t d")
        elif len(link_orig_shape) == 5:  # batch, horizon, num_links, 4, 4  # homogeneous transform tensor
            b, h, t, d, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d d -> (b h) t d d")
        else:
            raise NotImplementedError

        # link_tensor_pos
        # position: (batch horizon) x num_links x 3
        cost = self.compute_costs_impl(q, link_pos, *args, **kwargs)

        if cost.ndim == 1:
            cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        # if len(link_orig_shape) == 4 or len(link_orig_shape) == 5:
        #     cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        return cost

    @abstractmethod
    def compute_costs_impl(self, *args, **kwargs):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


def interpolate_points_v1(points, num_interpolated_points):
    # https://github.com/SamsungLabs/RAMP/blob/c3bd23b2c296c94cdd80d6575390fd96c4f83d83/mppi_planning/cost/collision_cost.py#L89
    points = Functional.interpolate(points.transpose(-2, -1), size=num_interpolated_points, mode='linear', align_corners=True).transpose(-2, -1)
    return points


# Old implementation
def interpolate_points_v2(points, num_interpolate, link_interpolate_range):
    if num_interpolate > 0:
        link_dim = points.shape[:-1]
        alpha = torch.linspace(0, 1, num_interpolate + 2).type_as(points)[1:num_interpolate + 1]
        alpha = alpha.view(tuple([1] * len(link_dim) + [-1, 1]))  # 1 x 1 x 1 x ... x num_interpolate x 1
        X = points[..., link_interpolate_range[0]:link_interpolate_range[1] + 1, :].unsqueeze(-2)  # batch_dim x num_interp_link x 1 x 3
        X_diff = torch.diff(X, dim=-3)  # batch_dim x (num_interp_link - 1) x 1 x 3
        X_interp = X[..., :-1, :, :] + X_diff * alpha  # batch_dim x (num_interp_link - 1) x num_interpolate x 3
        points = torch.cat([points, X_interp.flatten(-3, -2)], dim=-2)  # batch_dim x (num_link + (num_interp_link - 1) * num_interpolate) x 3
    return points


class EmbodimentDistanceFieldBase(DistanceField):

    def __init__(self,
                 robot,
                 link_idxs_for_collision_checking=None,
                 num_interpolated_points=30,
                 collision_margins=0.,
                 cutoff_margin=0.001,
                 field_type='sdf', clamp_sdf=True,
                 interpolate_link_pos=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert robot is not None, "You need to pass a robot instance to the embodiment distance fields"
        self.robot = robot
        self.link_idxs_for_collision_checking = link_idxs_for_collision_checking
        self.num_interpolated_points = num_interpolated_points
        self.collision_margins = collision_margins
        self.cutoff_margin = cutoff_margin
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.interpolate_link_pos = interpolate_link_pos

    def compute_embodiment_cost(self, q, link_pos, field_type=None, **kwargs):  # position tensor
        if field_type is None:
            field_type = self.field_type
        if field_type == 'rbf':
            return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
        elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
            margin = self.collision_margins + self.cutoff_margin
            # returns all distances from each link to the environment
            margin_minus_sdf = -(self.compute_embodiment_signed_distances(q, link_pos, **kwargs) - margin)
            if self.clamp_sdf:
                clamped_sdf = torch.relu(margin_minus_sdf)
            else:
                clamped_sdf = margin_minus_sdf
            if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
                clamped_sdf = clamped_sdf.max(-2)[0]
            # sum over link points for gradient computation
            return clamped_sdf.sum(-1)
        elif field_type == 'occupancy':
            return self.compute_embodiment_collision(q, link_pos, **kwargs)
            # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
            # return (distances < margin).sum(-1)
        else:
            raise NotImplementedError('field_type {} not implemented'.format(field_type))

    # def compute_embodiment_cost(self, q, link_pos, field_type=None, **kwargs):  # position tensor
    #     if field_type is None:
    #         field_type = self.field_type
    #     if field_type == 'rbf':
    #         return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
    #     elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
    #         margin = self.collision_margins + self.cutoff_margin
    #         # computes sdf of all links to all objects (or sdf map)
    #         margin_minus_sdf = -(self.compute_embodiment_signed_distances(q, link_pos, **kwargs) - margin)
    #         if self.clamp_sdf:
    #             clamped_sdf = torch.relu(margin_minus_sdf)
    #         else:
    #             clamped_sdf = margin_minus_sdf
    #
    #         if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
    #             clamped_sdf = clamped_sdf.max(-2)[0]
    #
    #         # sum over link points for gradient computation
    #         return clamped_sdf.sum(-1)
    #
    #     elif field_type == 'occupancy':
    #         return self.compute_embodiment_collision(q, link_pos, **kwargs)
    #         # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
    #         # return (distances < margin).sum(-1)
    #     else:
    #         raise NotImplementedError('field_type {} not implemented'.format(field_type))

    def compute_costs_impl(self, q, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x 3
        # interpolate to approximate link spheres
        if self.robot.grasped_object is not None:
            n_grasped_object_points = self.robot.grasped_object.n_base_points_for_collision
            link_pos_robot = link_pos[..., :-n_grasped_object_points, :]
            link_pos_grasped_object = link_pos[..., -n_grasped_object_points:, :]
        else:
            link_pos_robot = link_pos

        link_pos_robot = link_pos_robot[..., self.link_idxs_for_collision_checking, :]
        if self.interpolate_link_pos:
            # select the robot links used for collision checking
            link_pos = interpolate_points_v1(link_pos_robot, self.num_interpolated_points)
        else:
            link_pos = link_pos_robot

        # stack collision points from grasped object
        # these points do not need to be interpolated
        if self.robot.grasped_object is not None:
            link_pos = torch.cat((link_pos, link_pos_grasped_object), dim=-2)

        embodiment_cost = self.compute_embodiment_cost(q, link_pos, **kwargs)
        return embodiment_cost

    def compute_distance(self, q, link_pos, **kwargs):
        raise NotImplementedError
        link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points)
        self_distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs).min(-1)[0]  # batch_dim
        return self_distances

    def zero_grad(self):
        pass
        # raise NotImplementedError

    @abstractmethod
    def compute_embodiment_rbf_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_collision(self, *args, **kwargs):
        raise NotImplementedError


class CollisionSelfField(EmbodimentDistanceFieldBase):

    def __init__(self, *args, idxs_links_distance_matrix=None, **kwargs):
        super().__init__(*args, collision_margins=0., **kwargs)
        self.idxs_links_distance_matrix = idxs_links_distance_matrix
        self.idxs_links_distance_matrix_tuple = tuple(zip(*idxs_links_distance_matrix))

    def compute_embodiment_rbf_distances(self, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get('margin', self.cutoff_margin)
        rbf_distance = torch.exp(torch.square(link_pos.unsqueeze(-2) -
                                              link_pos.unsqueeze(-3)).sum(-1) / (-margin ** 2 * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):  # position tensor
        if link_pos.shape[-2] == 1:
            # if there is only one link, the self distance is very large
            # implementation guarantees gradient computation
            return torch.abs(link_pos).sum(-1) * 1e9
        dist_mat = torch.linalg.norm(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3), dim=-1)  # batch_dim x links x links

        # select lower triangular -- distance between link points
        # lower_indices = torch.tril_indices(dist_mat.shape[-1], dist_mat.shape[-1], offset=-1).unbind()
        # distances = dist_mat[..., lower_indices[0], lower_indices[1]]  # batch_dim x (links * (links - 1) / 2)

        # select only distances between pairs of specified links
        distances = dist_mat[..., self.idxs_links_distance_matrix_tuple[0], self.idxs_links_distance_matrix_tuple[1]]

        return distances

    def compute_embodiment_collision(self, q, link_pos, **kwargs):  # position tensor
        margin = kwargs.get('margin', self.cutoff_margin)
        distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs)
        any_self_collision = torch.any(distances < margin, dim=-1)
        return any_self_collision


def reshape_q(q):
    q_orig_shape = q.shape
    if len(q_orig_shape) == 2:
        h = 1
        b, d = q_orig_shape
    elif len(q_orig_shape) == 3:
        b, h, d = q_orig_shape
        q = einops.rearrange(q, "b h d -> (b h) d")
    else:
        raise NotImplementedError
    return q, q_orig_shape, b, h, d


class CollisionSelfFieldWrapperSTORM(EmbodimentDistanceFieldBase):

    def __init__(self, robot, weights_fname, n_joints, *args, **kwargs):
        super().__init__(robot, *args, collision_margins=0., interpolate_link_pos=False, **kwargs)
        self.robot_self_collision_net = RobotSelfCollisionNet(n_joints)
        self.robot_self_collision_net.load_weights(weights_fname, self.tensor_args)

    def compute_costs_impl(self, q, link_pos, **kwargs):
        embodiment_cost = self.compute_embodiment_cost(q, link_pos, **kwargs)
        return embodiment_cost

    def compute_embodiment_rbf_distances(self, *args, **kwargs):  # position tensor
        raise NotImplementedError

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):  # position tensor
        q, q_orig_shape, b, h, d = reshape_q(q)

        # multiply by -1, because according to the paper (page 6, https://arxiv.org/pdf/2104.13542.pdf)
        # "Distance is positive when two links are penetrating and negative when not colliding"
        sdf_self_collision = -1. * self.robot_self_collision_net.compute_signed_distance(q, with_grad=True)

        return sdf_self_collision

    def compute_embodiment_collision(self, q, link_pos, **kwargs):  # position tensor
        q, q_orig_shape, b, h, d = reshape_q(q)

        # multiply by -1, because according to the paper (page 6, https://arxiv.org/pdf/2104.13542.pdf)
        # "Distance is positive when two links are penetrating and negative when not colliding"
        sdf_self_collision = -1. * self.robot_self_collision_net.compute_signed_distance(q, with_grad=True)

        if len(q_orig_shape) == 3:
            sdf_self_collision = einops.rearrange(sdf_self_collision, "(b h) 1 -> b h", b=b, h=h)

        any_self_collision = sdf_self_collision < -0.05  # trained on 0.02
        return any_self_collision


class CollisionObjectBase(EmbodimentDistanceFieldBase):

    def __init__(self, *args, link_margins_for_object_collision_checking_tensor=None, **kwargs):
        super().__init__(*args, collision_margins=link_margins_for_object_collision_checking_tensor, **kwargs)

    def compute_embodiment_rbf_distances(self, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get('margin', self.margin)
        rbf_distance = torch.exp(torch.square(self.object_signed_distances(link_pos, **kwargs)) / (-margin ** 2 * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):
        return self.object_signed_distances(link_pos, **kwargs)

    def compute_embodiment_collision(self, q, link_pos, **kwargs):
        # position tensor
        margin = kwargs.get('margin', self.collision_margins + self.cutoff_margin)
        signed_distances = self.object_signed_distances(link_pos, **kwargs)
        collisions = signed_distances < margin
        # reduce over points (dim -1) and over objects (dim -2)
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, *args, **kwargs):
        raise NotImplementedError


class CollisionObjectDistanceField(CollisionObjectBase):

    def __init__(self,
                 *args,
                 df_obj_list_fn=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.df_obj_list_fn = df_obj_list_fn

    def object_signed_distances(self, link_pos, **kwargs):
        if self.df_obj_list_fn is None:
            return torch.inf
        df_obj_list = self.df_obj_list_fn()
        link_dim = link_pos.shape[:-1]
        link_pos = link_pos.reshape(-1, link_pos.shape[-1])  # flatten batch_dim and links
        dfs = []
        for df in df_obj_list:
            dfs.append(df.compute_signed_distance(link_pos).view(link_dim))  # df() returns batch_dim x links
        return torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links


class CollisionWorkspaceBoundariesDistanceField(CollisionObjectBase):

    def __init__(self, *args, ws_bounds_min=None, ws_bounds_max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_min = ws_bounds_min
        self.ws_max = ws_bounds_max

    def object_signed_distances(self, link_pos, **kwargs):
        signed_distances_bounds_min = link_pos - self.ws_min
        signed_distances_bounds_min = torch.sign(signed_distances_bounds_min) * torch.abs(signed_distances_bounds_min)
        signed_distances_bounds_max = self.ws_max - link_pos
        signed_distances_bounds_max = torch.sign(signed_distances_bounds_max) * torch.abs(signed_distances_bounds_max)
        signed_distances_bounds = torch.cat((signed_distances_bounds_min, signed_distances_bounds_max), dim=-1)
        return signed_distances_bounds.transpose(-2, -1)    # batch_dim x num_sdfs x links


class EESE3DistanceField(DistanceField):

    def __init__(self, target_H, w_pos=1., w_rot=1., square=True, **kwargs):
        super().__init__(**kwargs)
        self.target_H = target_H
        self.square = square
        self.w_pos = w_pos
        self.w_rot = w_rot

    def update_target(self, target_H):
        self.target_H = target_H

    def compute_distance(self, link_tensor):
        # homogeneous transformation link_tensor # batch x num_links x 4 x 4
        # -1: get EE as last link  # TODO - get EE from its name id
        return SE3_distance(link_tensor[..., -1, :, :], self.target_H, w_pos=self.w_pos, w_rot=self.w_rot)

    def compute_costs_impl(self, q, link_tensor, **kwargs):  # position tensor
        dist = self.compute_distance(link_tensor).squeeze()
        if self.square:
            dist = torch.square(dist)
        return dist

    def zero_grad(self):
        raise NotImplementedError




########################################################################################################################
# DEPRECATED! NEEDS REVISION
########################################################################################################################
#
# class FloorDistanceField(DistanceField):
#
#     def __init__(self, margin=0.05, **kwargs):
#         super().__init__(**kwargs)
#         self.margin = margin
#
#     def distances(self, link_tensor):
#         link_pos = link_tensor[..., :3, -1]
#         return link_pos[..., 2]  # z axis
#
#     def compute_collision(self, link_tensor, floor_min=None):
#         collisions = torch.zeros(link_tensor.shape[:2]).to(**self.tensor_args)  # batch, trajectory
#         if floor_min is None:
#             return collisions
#         distances = self.distances(link_tensor)
#         floor_collisions = distances < floor_min
#         any_floor_collisions = torch.any(floor_collisions, dim=-1)
#         return any_floor_collisions
#
#     def compute_distance(self, link_tensor):
#         distances = self.distances(link_tensor)
#         return distances.mean(-1)  # z axis
#
#     def compute_cost(self, link_tensor, **kwargs):
#         return torch.exp(-0.5 * torch.square(link_tensor[..., 2, -1].mean(-1)) / self.margin ** 2)
#
#     def zero_grad(self):
#         raise NotImplementedError
#
#
# class BorderDistanceFieldOLD(DistanceField):
#
#     def __init__(self, target_H, w_pos=1., w_rot=1., square=True, **kwargs):
#         super().__init__(**kwargs)
#         self.target_H = target_H
#         self.square = square
#         self.w_pos = w_pos
#         self.w_rot = w_rot
#
#     def update_target(self, target_H):
#         self.target_H = target_H
#
#     def distances(self, link_tensor):
#         link_pos = link_tensor[..., :3, -1]
#         return link_pos
#
#     def compute_collision(self, link_tensor, task_space_min=None, task_space_max=None):
#         collisions = torch.zeros(link_tensor.shape[:2]).to(**self.tensor_args)  # batch, trajectory
#         if task_space_min is None and task_space_max is None:
#             return collisions
#         distances = self.distances(link_tensor)
#         collisions = torch.logical_or(distances < task_space_min, distances > task_space_max)
#         any_collisions = torch.any(torch.any(collisions, dim=-1), dim=-1)
#         return any_collisions
#
#     def compute_distance(self, link_tensor):
#         raise NotImplementedError
#         distances = self.distances(link_tensor)
#         return distances.mean(-1)  # z axis
#
#     def compute_cost(self, link_tensor, **kwargs):
#         raise NotImplementedError
#         return torch.exp(-0.5 * torch.square(link_tensor[..., 2, -1].mean(-1)) / self.margin ** 2)
#
#     def zero_grad(self):
#         raise NotImplementedError
#
#
# class SphereDistanceField(DistanceField):
#     """ This class holds a batched collision model where the robot is represented as spheres.
#         All points are stored in the world reference frame, obtained by using update_pose calls.
#     """
#
#     def __init__(self, robot_collision_params, batch_size=1, **kwargs):
#         """ Initialize with robot collision parameters, look at franka_reacher.py for an example.
#         Args:
#             robot_collision_params (Dict): collision model parameters
#             batch_size (int, optional): Batch size of parallel sdf computation. Defaults to 1.
#             tensor_args (dict, optional): compute device and data type. Defaults to {'device':"cpu", 'dtype':torch.float32}.
#         """
#         super().__init__(**kwargs)
#
#         self.batch_dim = [batch_size, ]
#
#         self._link_spheres = None
#         self._batch_link_spheres = None
#         self.w_batch_link_spheres = None
#
#         self.robot_collision_params = robot_collision_params
#         self.load_robot_collision_model(robot_collision_params)
#         self.margin = robot_collision_params.get('margin', 0.1)
#
#         self.self_dist = None
#         self.obst_dist = None
#
#     def load_robot_collision_model(self, robot_collision_params):
#         """Load robot collision model, called from constructor
#         Args:
#             robot_collision_params (Dict): loaded from yml file
#         """
#         self.robot_links = robot_collision_params['link_objs']
#
#         # load collision file:
#         coll_yml = (get_configs_path() / robot_collision_params['collision_spheres']).as_posix()
#         with open(coll_yml) as file:
#             coll_params = yaml.load(file, Loader=yaml.FullLoader)
#
#         self._link_spheres = []
#         # we store as [n_link, n_dim]
#
#         for j_idx, j in enumerate(self.robot_links):
#             n_spheres = len(coll_params[j])
#             link_spheres = torch.zeros((n_spheres, 4)).to(**self.tensor_args)
#             for i in range(n_spheres):
#                 link_spheres[i, :] = tensor_sphere(coll_params[j][i][:3], coll_params[j][i][3], tensor=link_spheres[i]).to(**self.tensor_args)
#             self._link_spheres.append(link_spheres)
#
#     def build_batch_features(self, clone_objs=False, batch_dim=None):
#         """clones poses/object instances for computing across batch.
#         Use this once per batch size change to avoid re-initialization over repeated calls.
#         Args:
#             clone_objs (bool, optional): clones objects. Defaults to False.
#             batch_size ([type], optional): batch_size to clone. Defaults to None.
#         """
#         if batch_dim is not None:
#             self.batch_dim = batch_dim
#         if clone_objs:
#             self._batch_link_spheres = []
#             for i in range(len(self._link_spheres)):
#                 _batch_link_i = self._link_spheres[i].view(
#                     tuple([1] * len(self.batch_dim) + list(self._link_spheres[i].shape)))
#                 _batch_link_i = _batch_link_i.repeat(tuple(self.batch_dim + [1, 1]))
#                 self._batch_link_spheres.append(_batch_link_i)
#         self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres)
#
#     def update_batch_robot_collision_objs(self, links_dict):
#         '''update pose of link spheres
#         Args:
#         links_pos: bxnx3
#         links_rot: bxnx3x3
#         '''
#         for i in range(len(self.robot_links)):
#             link_H = links_dict[self.robot_links[i]].get_transform_matrix()
#             link_pos, link_rot = link_H[..., :-1, -1], link_H[..., :3, :3]
#             self.w_batch_link_spheres[i][..., :3] = transform_point(
#                 self._batch_link_spheres[i][..., :3], link_rot, link_pos.unsqueeze(-2)
#             )
#
#     def check_collisions(self, obstacle_spheres=None):
#         """Analytic method to compute signed distance between links.
#         Args:
#             link_trans ([tensor]): link translation as batch [b, 3]
#             link_rot ([type]): link rotation as batch [b, 3, 3]
#         Returns:
#             [tensor]: signed distance [b, 1]
#         """
#         n_links = len(self.w_batch_link_spheres)
#         if self.self_dist is None:
#             self.self_dist = torch.zeros(self.batch_dim + [n_links, n_links], device=self.device) - 100.0
#         if self.obst_dist is None:
#             self.obst_dist = torch.zeros(self.batch_dim + [n_links, ], device=self.device) - 100.0
#         dist = self.self_dist
#         dist = find_link_distance(self.w_batch_link_spheres, dist)
#         total_dist = dist.max(1)[0]
#         if obstacle_spheres is not None:
#             obst_dist = self.obst_dist
#             obst_dist = find_obstacle_distance(obstacle_spheres, self.w_batch_link_spheres, obst_dist)
#             total_dist += obst_dist
#         return dist
#
#     def compute_distance(self, links_dict, obstacle_spheres=None):
#         self.update_batch_robot_collision_objs(links_dict)
#         return self.check_collisions(obstacle_spheres).max(1)[0]
#
#     def compute_cost(self, links_dict, obstacle_spheres=None):
#         signed_dist = self.compute_distance(links_dict, obstacle_spheres=obstacle_spheres)
#         return torch.exp(signed_dist / self.margin)
#
#     def get_batch_robot_link_spheres(self):
#         return self.w_batch_link_spheres
#
#     def zero_grad(self):
#         raise NotImplementedError
#         self.self_dist.detach_()
#         self.self_dist.grad = None
#         self.obst_dist.detach_()
#         self.obst_dist.grad = None
#         for i in range(len(self.robot_links)):
#             self.w_batch_link_spheres[i].detach_()
#             self.w_batch_link_spheres[i].grad = None
#             self._batch_link_spheres[i].detach_()
#             self._batch_link_spheres[i].grad = None
#
#
# class LinkDistanceField(DistanceField):
#
#     def __init__(self, field_type='sdf', clamp_sdf=False, num_interpolate=0, link_interpolate_range=[5, 7],
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.field_type = field_type
#         self.clamp_sdf = clamp_sdf
#         self.num_interpolate = num_interpolate
#         self.link_interpolate_range = link_interpolate_range
#
#     def distances(self, link_tensor, obstacle_spheres):
#         link_pos = link_tensor[..., :3, -1]
#         link_pos = link_pos.unsqueeze(-2)
#         obstacle_spheres = obstacle_spheres.unsqueeze(0).unsqueeze(0)
#         centers = obstacle_spheres[..., :3]
#         radii = obstacle_spheres[..., 3]
#         return torch.linalg.norm(link_pos - centers, dim=-1) - radii
#
#     def compute_collision(self, link_tensor, obstacle_spheres=None, buffer=0.02):  # position tensor
#         collisions = torch.zeros(link_tensor.shape[:2]).to(**self.tensor_args)  # batch, trajectory
#         if obstacle_spheres is None:
#             return collisions
#         distances = self.distances(link_tensor, obstacle_spheres)
#         collisions = torch.any(torch.any(distances < buffer, dim=-1), dim=-1)
#         return collisions
#
#     def compute_distance(self, link_tensor, obstacle_spheres=None, **kwargs):
#         if obstacle_spheres is None:
#             return 1e10
#         link_tensor = link_tensor[..., :3, -1].unsqueeze(-2)
#         obstacle_spheres = obstacle_spheres.unsqueeze(0)
#         return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) - obstacle_spheres[..., 3]).sum((-1, -2))
#
#     def compute_cost(self, link_tensor, obstacle_spheres=None, **kwargs):
#         if obstacle_spheres is None:
#             return 0
#         link_tensor = link_tensor[..., :3, -1]
#         link_dim = link_tensor.shape[:-1]
#         if self.num_interpolate > 0:
#             alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
#             alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
#             for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
#                 X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
#                 eval_sphere = X1 + (X2 - X1) * alpha
#                 link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
#         link_tensor = link_tensor.unsqueeze(-2)
#         obstacle_spheres = obstacle_spheres.unsqueeze(0)
#         # signed distance field
#         if self.field_type == 'rbf':
#             return torch.exp(-0.5 * torch.square(link_tensor - obstacle_spheres[..., :3]).sum(-1) / torch.square(obstacle_spheres[..., 3])).sum((-1, -2))
#         elif self.field_type == 'sdf':
#             sdf = -torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) + obstacle_spheres[..., 3]
#             if self.clamp_sdf:
#                 sdf = sdf.clamp(max=0.)
#             return sdf.max(-1)[0].max(-1)[0]
#         elif self.field_type == 'occupancy':
#             return (torch.linalg.norm(link_tensor - obstacle_spheres[..., :3], dim=-1) < obstacle_spheres[..., 3]).sum((-1, -2))
#
#     def zero_grad(self):
#         raise NotImplementedError
#
#
# class LinkSelfDistanceField(DistanceField):
#
#     def __init__(self, margin=0.03, num_interpolate=0, link_interpolate_range=[5, 7], **kwargs):
#         super().__init__(**kwargs)
#         self.num_interpolate = num_interpolate
#         self.link_interpolate_range = link_interpolate_range
#         self.margin = margin
#
#     def distances(self, link_tensor):
#         link_pos = link_tensor[..., :3, -1]
#         return torch.linalg.norm(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3), dim=-1)
#
#     def compute_collision(self, link_tensor, buffer=0.05):  # position tensor
#         distances = self.distances(link_tensor)
#         self_collisions = torch.tril(distances < buffer, diagonal=-2)
#         any_self_collision = torch.any(torch.any(self_collisions, dim=-1), dim=-1)
#         return any_self_collision
#
#     def compute_distance(self, link_tensor):  # position tensor
#         distances = self.distances(link_tensor)
#         return distances.sum((-1, -2))
#
#     def compute_cost(self, link_tensor, **kwargs):   # position tensor
#         link_tensor = link_tensor[..., :3, -1]
#         link_dim = link_tensor.shape[:-1]
#         if self.num_interpolate > 0:
#             alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
#             alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
#             for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
#                 X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
#                 eval_sphere = X1 + (X2 - X1) * alpha
#                 link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
#         return torch.exp(torch.square(link_tensor.unsqueeze(-2) - link_tensor.unsqueeze(-3)).sum(-1) / (-self.margin**2 * 2)).sum((-1, -2))
#
#     def zero_grad(self):
#         raise NotImplementedError
#
#
#
#
#
# class SkeletonSE3DistanceField(DistanceField):
#
#     def __init__(self, target_H=None, w_pos=1., w_rot=1., link_list=None, square=True, **kwargs):
#         super().__init__(**kwargs)
#         self.target_H = target_H
#         self.w_pos = w_pos
#         self.w_rot = w_rot
#         self.square = square
#         self.link_list = link_list
#         self.link_weights = None
#         self.link_weights_ts = None
#
#     def set_link_weights(self, weight_dict):
#         assert all(link in self.link_list for link in weight_dict)
#         self.link_weights.update(weight_dict)
#         total = sum(self.link_weights.values())
#         weights = np.array([self.link_weights[l] / total for l in self.link_weights])
#         self.link_weights_ts = torch.from_numpy(weights).to(self.device)
#
#     def construct_target_from_configuration(self, model, q, link_list=None):
#         self.device = model._device
#         if isinstance(q, np.ndarray) or isinstance(q, list):
#             q = torch.tensor(q, device=self.device)
#         if q.ndim == 1:
#             q = q.unsqueeze(0)
#         self.target_H = model.compute_forward_kinematics_link_list(q, return_dict=False, link_list=link_list)
#         if link_list is None:
#             if model.link_list is None:
#                 self.link_list = model.get_link_names()
#             else:
#                 self.link_list = model.link_list
#         else:
#             self.link_list = link_list
#         self.link_weights = {}
#         for l in self.link_list:
#             self.link_weights[l] = 1.
#         self.link_weights_ts = torch.ones(len(self.link_list), device=self.device) / len(self.link_list)
#
#     def compute_distance(self, link_tensor, **kwargs):  # assume link_tensor has index ordering like link_list
#         link_list = kwargs.get('link_list', None)
#         link_weights_ts = self.link_weights_ts
#         target_H = self.target_H
#         if link_list is not None:
#             link_indices = [self.link_list.index(l) for l in link_list]
#             link_tensor = link_tensor[..., link_indices, :, :]
#             target_H = target_H[..., link_indices, :, :]
#             link_weights_ts = link_weights_ts[link_indices] * len(self.link_list) / len(link_indices)  # reweight
#         return (SE3_distance(link_tensor, target_H, w_pos=self.w_pos, w_rot=self.w_rot) * link_weights_ts).sum(-1)
#
#     def compute_cost(self, link_tensor, **kwargs):  # position tensor
#         dist = self.compute_distance(link_tensor, **kwargs).squeeze()
#         if self.square:
#             dist = torch.square(dist)
#         return dist
#
#     def zero_grad(self):
#         raise NotImplementedError
#
#
# class MeshDistanceField(DistanceField):
#
#     def __init__(self, mesh_file, margin=0.03, field_type='sdf', num_interpolate=0, link_interpolate_range=[], base_position=None, base_orientation=None, tensor_args=None) -> None:
#         if tensor_args is None:
#             tensor_args = dict(device='cpu', dtype=torch.float32)
#         self.tensor_args = tensor_args
#         self.field_type = field_type
#         self.num_interpolate = num_interpolate
#         self.link_interpolate_range = link_interpolate_range
#         self.margin = margin
#         if base_position is None:
#             base_position = torch.zeros(3, **self.tensor_args)
#         else:
#             base_position = torch.tensor(base_position, **self.tensor_args)
#         if base_orientation is None:
#             base_orientation = torch.eye(3, **self.tensor_args)
#         else:
#             base_orientation = torch.tensor(base_orientation, **self.tensor_args)
#
#         # Load mesh
#         self.mesh = trimesh.load_mesh(mesh_file)
#         if isinstance(self.mesh, trimesh.Scene):
#             self.mesh = self.mesh.dump()[0]
#         self.vertices = torch.tensor(self.mesh.vertices, **self.tensor_args)
#         self.faces = torch.tensor(self.mesh.faces, **self.tensor_args)
#         # Transform mesh
#         self.vertices = self.vertices @ base_orientation.T + base_position
#
#     def compute_distance(self, points, sample=False, num_samples=100):
#         if sample:
#             indices = np.random.choice(self.vertices.shape[0], num_samples, replace=False)
#             vertices = self.vertices[indices]
#         else:
#             vertices = self.vertices
#         distances = torch.min(torch.norm(points.unsqueeze(-2) - vertices, dim=-1), dim=-1)[0]
#         return distances
#
#     def compute_cost(self, link_tensor, sample=False, num_samples=100, **kwargs):
#         if sample:
#             indices = np.random.choice(self.vertices.shape[0], num_samples, replace=False)
#             vertices = self.vertices[indices]
#         else:
#             vertices = self.vertices
#         link_tensor = link_tensor[..., :3, -1]
#         link_dim = link_tensor.shape[:-1]
#         if self.num_interpolate > 0:
#             alpha = torch.linspace(0, 1, self.num_interpolate + 2).type_as(link_tensor)[1:self.num_interpolate + 1]
#             alpha = alpha.view(tuple([1] * (len(link_dim) - 1) + [-1, 1]))
#             for i in range(self.link_interpolate_range[0], self.link_interpolate_range[1]):
#                 X1, X2 = link_tensor[..., i, :].unsqueeze(-2), link_tensor[..., i + 1, :].unsqueeze(-2)
#                 eval_sphere = X1 + (X2 - X1) * alpha
#                 link_tensor = torch.cat([link_tensor, eval_sphere], dim=-2)
#         link_tensor = link_tensor.unsqueeze(-2)
#         vertices = vertices.unsqueeze(0)
#         # signed distance field
#         if self.field_type == 'rbf':
#             return torch.exp(-0.5 * torch.square(link_tensor - vertices[..., :3]).sum(-1) / (self.margin ** 2)).sum((-1, -2))
#         elif self.field_type == 'occupancy':
#             return (torch.linalg.norm(link_tensor - vertices[..., :3], dim=-1) < self.margin[..., 3]).sum((-1, -2))
#
#     def zero_grad(self):
#         raise NotImplementedError



if __name__ == '__main__':
    raise NotImplementedError
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
