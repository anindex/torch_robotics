import sys
from abc import ABC

import einops
import numpy as np
import torch

from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionWorkspaceBoundariesDistanceField, \
    CollisionSelfField, CollisionObjectDistanceField
from torch_robotics.trajectory.utils import interpolate_traj_via_points


class Task(ABC):

    def __init__(self, env=None, robot=None, tensor_args=None, **kwargs):
        self.env = env
        self.robot = robot
        self.tensor_args = tensor_args


class PlanningTask(Task):

    def __init__(
            self,
            ws_limits=None,
            use_occupancy_map=False,
            cell_size=0.01,
            obstacle_buffer=0.01,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ws_limits = self.env.limits if ws_limits is None else ws_limits
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

        # Optional: use an occupancy map for collision checking -- useful for sampling-based algorithms
        # A precomputed collision map is faster when checking for collisions, in comparison to computing the distances
        # from task spaces to objects
        self.use_occupancy_map = use_occupancy_map
        if use_occupancy_map:
            self.env.build_occupancy_map(cell_size=cell_size)

        ################################################################################################
        # Collision fields
        self.obstacle_buffer = obstacle_buffer
        # collision field for self-collision
        self.df_collision_self = CollisionSelfField(
            num_interpolate=self.robot.num_interpolate,
            link_interpolate_range=self.robot.link_interpolate_range,
            margin=self.robot.self_collision_margin,
            tensor_args=self.tensor_args
        )

        # collision field for objects
        self.df_collision_objects = CollisionObjectDistanceField(
            num_interpolate=self.robot.num_interpolate,
            link_interpolate_range=self.robot.link_interpolate_range,
            margin=obstacle_buffer,
            tensor_args=self.tensor_args
        )

        # collision field for workspace boundaries
        self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
            num_interpolate=self.robot.num_interpolate,
            link_interpolate_range=self.robot.link_interpolate_range,
            margin=0.,
            ws_bounds_min=self.ws_min,
            ws_bounds_max=self.ws_max,
            tensor_args=self.tensor_args
        )

    def distance_q(self, q1, q2):
        return self.robot.distance_q(q1, q2)

    def sample_q(self, without_collision=True, **kwargs):
        if without_collision:
            return self.random_coll_free_q(**kwargs)
        else:
            return self.robot.random_q(**kwargs)

    def random_coll_free_q(self, n_samples=1, max_samples=1000, max_tries=1000):
        # Random position in configuration space not in collision
        reject = True
        samples = torch.zeros((n_samples, self.robot.q_dim), **self.tensor_args)
        idx_begin = 0
        for i in range(max_tries):
            qs = self.robot.random_q(max_samples)
            in_collision = self.compute_collision(qs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            if idxs_not_in_collision.nelement() == 1:
                idxs_not_in_collision = [idxs_not_in_collision]
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()

    def compute_collision(self, x, **kwargs):
        q_pos = self.robot.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type='occupancy', **kwargs)

    def compute_collision_cost(self, x, **kwargs):
        q_pos = self.robot.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type='sdf', **kwargs)

    def _compute_collision_or_cost(self, q, field_type='occupancy', **kwargs):
        # q.shape needs to be reshaped to (batch, horizon, q_dim)
        q_original_shape = q.shape
        b = 1
        h = 1
        collisions = None
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)  # add batch and horizon dimensions for interface
            collisions = torch.ones((1, ), **self.tensor_args)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(1)  # add horizon dimension for interface
            collisions = torch.ones((b, 1), **self.tensor_args)  # (batch, 1)
        elif q.ndim == 3:
            b = q.shape[0]
            h = q.shape[1]
            collisions = torch.ones((b, h), **self.tensor_args)  # (batch, horizon)
        elif q.ndim > 3:
            raise NotImplementedError

        if self.use_occupancy_map:
            # ---------------------------------- For occupancy maps ----------------------------------
            ########################################
            # Configuration space boundaries
            idxs_coll_free = torch.argwhere(torch.all(
                torch.logical_and(torch.greater_equal(q, self.robot.q_min), torch.less_equal(q, self.robot.q_max)),
                dim=-1))  # I, 2

            # check if all points are out of bounds (in collision)
            if idxs_coll_free.nelement() == 0:
                return collisions

            ########################################
            # Task space collisions
            # forward kinematics
            q_try = q[idxs_coll_free[:, 0], idxs_coll_free[:, 1]]  # I, q_dim
            x_pos = self.robot.fk_map(q_try, pos_only=True)  # I, taskspaces, x_dim

            # workspace boundaries
            # configuration is not valid if any points in the task spaces is out of workspace boundaries
            idxs_ws_in_boundaries = torch.argwhere(torch.all(torch.all(torch.logical_and(
                torch.greater_equal(x_pos, self.ws_min), torch.less_equal(x_pos, self.ws_max)), dim=-1),
                dim=-1)).squeeze()  # I_ws

            idxs_coll_free = idxs_coll_free[idxs_ws_in_boundaries].view(-1, 2)

            # collision in task space
            x_pos_in_ws = x_pos[idxs_ws_in_boundaries]  # I_ws, x_dim
            collisions_pos_x = self.env.occupancy_map.get_collisions(x_pos_in_ws, **kwargs)
            if len(collisions_pos_x.shape) == 1:
                collisions_pos_x = collisions_pos_x.view(1, -1)
            idxs_taskspace = torch.argwhere(torch.all(collisions_pos_x == 0, dim=-1)).squeeze()

            idxs_coll_free = idxs_coll_free[idxs_taskspace].view(-1, 2)

            # filter collisions
            if len(collisions) == 1:
                collisions[idxs_coll_free[:, 0]] = 0
            else:
                collisions[idxs_coll_free[:, 0], idxs_coll_free[:, 1]] = 0
        else:
            # ---------------------------------- For distance fields ----------------------------------
            ########################################
            # For distance fields

            # forward kinematics
            x_pos = self.robot.fk_map(q, pos_only=True)  # batch, horizon, taskspaces, x_dim

            ########################
            # Self collision
            cost_collision_self = self.df_collision_self.compute_cost(
                x_pos, df_list=self.env.obj_list, field_type=field_type)

            # Object collision
            cost_collision_objects = self.df_collision_objects.compute_cost(
                x_pos, df_list=self.env.obj_list, field_type=field_type)

            # Workspace boundaries
            cost_collision_border = self.df_collision_ws_boundaries.compute_cost(
                x_pos, field_type=field_type)

            if field_type == 'occupancy':
                collisions = cost_collision_self | cost_collision_objects | cost_collision_border
            else:
                collisions = cost_collision_self + cost_collision_objects + cost_collision_border

        return collisions

    def get_trajs_collision_and_free(self, trajs, return_indices=False, margin=0.001, num_interpolation=5):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:  # n_goals (or steps), batch of trajectories, length, dim
            N, B, H, D = trajs.shape
            trajs_new = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape
            trajs_new = trajs

        # compute collisions on a finer interpolated trajectory
        trajs_new = interpolate_traj_via_points(trajs_new, num_intepolation=num_interpolation)
        trajs_waypoints_collisions = self.compute_collision(trajs_new, margin=margin)

        if trajs.ndim == 4:
            trajs_waypoints_collisions = einops.rearrange(trajs_waypoints_collisions, '(N B) H -> N B H', N=N, B=B)

        trajs_free_idxs = torch.argwhere(torch.logical_not(trajs_waypoints_collisions).all(dim=-1))
        trajs_coll_idxs = torch.argwhere(trajs_waypoints_collisions.any(dim=-1))

        if trajs.ndim == 4:
            trajs_free = trajs[trajs_free_idxs[:, 0], trajs_free_idxs[:, 1], ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0).unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs[:, 0], trajs_coll_idxs[:, 1], ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0).unsqueeze(0)
        else:
            trajs_free = trajs[trajs_free_idxs.squeeze(), ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs.squeeze(), ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0)

        if trajs_coll.nelement() == 0:
            trajs_coll = None
        if trajs_free.nelement() == 0:
            trajs_free = None

        if return_indices:
            return trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions
        return trajs_coll, trajs_free

    def compute_fraction_free_trajs(self, trajs, **kwargs):
        # Compute the fractions of trajs that are collision free
        _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.get_trajs_collision_and_free(trajs, return_indices=True)
        n_trajs_free = trajs_free_idxs.nelement()
        n_trajs_coll = trajs_coll_idxs.nelement()
        return n_trajs_free/(n_trajs_free + n_trajs_coll)

    def compute_collision_intensity_trajs(self, trajs, **kwargs):
        # Compute the fraction of waypoints that are in collision
        _, _, _, _, trajs_waypoints_collisions = self.get_trajs_collision_and_free(trajs, return_indices=True)
        return torch.count_nonzero(trajs_waypoints_collisions)/trajs_waypoints_collisions.nelement()

    def compute_success_free_trajs(self, trajs, **kwargs):
        # If at least one trajectory is collision free, then we consider success
        _, trajs_free = self.get_trajs_collision_and_free(trajs)
        if trajs_free is not None:
            if trajs_free.nelement() >= 1:
                return 1
        return 0