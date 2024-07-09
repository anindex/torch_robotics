import sys
from abc import ABC
from functools import partial

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionWorkspaceBoundariesDistanceField, \
    CollisionSelfField, CollisionObjectDistanceField
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.plot_utils import create_fig_and_axes, create_animation_video, plot_multiline


class Task(ABC):

    def __init__(self, env=None, robot=None, tensor_args=None, **kwargs):
        self.env = env
        self.robot = robot
        self.tensor_args = tensor_args


class PlanningTask(Task):

    def __init__(
            self,
            task_planner=None,
            ws_limits=None,
            use_occupancy_map=False,
            cell_size=0.01,
            obstacle_cutoff_margin=0.01,
            margin_for_waypoint_collision_checking=0.0,
            use_field_collision_self=False,  # consider self collision
            use_field_collision_objects=True,  # consider object collision
            use_field_collision_ws_boundaries=True,  # consider workspace boundaries collision
            **kwargs
    ):
        super().__init__(**kwargs)

        self.planner = task_planner

        self.ws_limits = self.env.limits if ws_limits is None else ws_limits
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

        # Optional: use an occupancy map for collision checking -- useful for sampling-based algorithms
        # A precomputed collision map is faster when checking for collisions, in comparison to computing the distances
        # from tasks spaces to objects
        self.use_occupancy_map = use_occupancy_map
        if use_occupancy_map:
            self.env.build_occupancy_map(cell_size=cell_size)

        self.margin_for_waypoint_collision_checking = margin_for_waypoint_collision_checking

        ################################################################################################
        # Collision fields
        # collision field for self-collision
        self.df_collision_self = self.robot.df_collision_self

        # collision field for objects
        self.df_collision_objects = CollisionObjectDistanceField(
            self.robot,
            df_obj_list_fn=self.env.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.robot.link_object_collision_margins,
            cutoff_margin=obstacle_cutoff_margin,
            tensor_args=self.tensor_args
        )

        self.df_collision_extra_objects = None
        if self.env.obj_extra_list is not None:
            self.df_collision_extra_objects = CollisionObjectDistanceField(
                self.robot,
                df_obj_list_fn=partial(self.env.get_df_obj_list, return_extra_objects_only=True),
                link_margins_for_object_collision_checking_tensor=self.robot.link_object_collision_margins,
                cutoff_margin=obstacle_cutoff_margin,
                tensor_args=self.tensor_args
            )
            self._collision_fields_extra_objects = [self.df_collision_extra_objects]
        else:
            self._collision_fields_extra_objects = []

        # collision field for workspace boundaries
        self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
            self.robot,
            link_margins_for_object_collision_checking_tensor=self.robot.link_object_collision_margins,
            cutoff_margin=obstacle_cutoff_margin,
            ws_bounds_min=self.ws_min,
            ws_bounds_max=self.ws_max,
            tensor_args=self.tensor_args
        )

        # TODO - self collision is not implemented
        assert not use_field_collision_self, "Self collision currently not implemented"
        self.df_collision_self = self.df_collision_self if use_field_collision_self else None

        self.df_collision_objects = self.df_collision_objects if use_field_collision_objects else None

        self.df_collision_ws_boundaries = self.df_collision_ws_boundaries if use_field_collision_ws_boundaries else None

        self._collision_fields = [
            self.df_collision_self,
            self.df_collision_objects,
            self.df_collision_ws_boundaries,
        ]


        ################################################################################################
        # Visualization
        self.colors = {'collision': 'black', 'free': 'orange'}
        self.colors_robot = {'collision': 'black', 'free': 'darkorange'}
        self.cmaps = {'collision': 'Greys', 'free': 'Oranges'}
        self.cmaps_robot = {'collision': 'Greys', 'free': 'YlOrRd'}

    def get_collision_fields(self):
        return [field for field in self._collision_fields if field is not None]

    def get_collision_fields_extra_objects(self):
        return [field for field in self._collision_fields_extra_objects if field is not None]

    def get_collision_objects_field(self):
        return self.df_collision_objects

    def get_collision_extra_objects_field(self):
        return self.df_collision_extra_objects

    def get_collision_ws_boundaries_field(self):
        return self.df_collision_ws_boundaries

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

    def _compute_collision_or_cost(self, q_pos, field_type='occupancy', **kwargs):
        # q.shape needs to be reshaped to (batch, horizon, q_dim)
        q_original_shape = q_pos.shape
        b = 1
        h = 1
        collisions = None
        if q_pos.ndim == 1:
            q_pos = q_pos.unsqueeze(0).unsqueeze(0)  # add batch and horizon dimensions for interface
            collisions = torch.ones((1, ), **self.tensor_args)
        elif q_pos.ndim == 2:
            b = q_pos.shape[0]
            q = q_pos.unsqueeze(1)  # add horizon dimension for interface
            collisions = torch.ones((b, 1), **self.tensor_args)  # (batch, 1)
        elif q_pos.ndim == 3:
            b = q_pos.shape[0]
            h = q_pos.shape[1]
            collisions = torch.ones((b, h), **self.tensor_args)  # (batch, horizon)
        elif q_pos.ndim > 3:
            raise NotImplementedError

        if self.use_occupancy_map:
            raise NotImplementedError
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
            x_pos = self.robot.fk_map_collision(q_try, pos_only=True)  # I, taskspaces, x_dim

            # workspace boundaries
            # configuration is not valid if any points in the tasks spaces is out of workspace boundaries
            idxs_ws_in_boundaries = torch.argwhere(torch.all(torch.all(torch.logical_and(
                torch.greater_equal(x_pos, self.ws_min), torch.less_equal(x_pos, self.ws_max)), dim=-1),
                dim=-1)).squeeze()  # I_ws

            idxs_coll_free = idxs_coll_free[idxs_ws_in_boundaries].view(-1, 2)

            # collision in tasks space
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
            fk_collision_pos = self.robot.fk_map_collision(q_pos)  # batch, horizon, taskspaces, x_dim

            ########################
            # Self collision
            if self.df_collision_self is not None:
                cost_collision_self = self.df_collision_self.compute_cost(q_pos, fk_collision_pos, field_type=field_type, **kwargs)
            else:
                cost_collision_self = 0

            # Object collision
            if self.df_collision_objects is not None:
                cost_collision_objects = self.df_collision_objects.compute_cost(q_pos, fk_collision_pos, field_type=field_type, **kwargs)
            else:
                cost_collision_objects = 0

            # Workspace boundaries
            if self.df_collision_ws_boundaries is not None:
                cost_collision_border = self.df_collision_ws_boundaries.compute_cost(q_pos, fk_collision_pos, field_type=field_type, **kwargs)
            else:
                cost_collision_border = 0

            if field_type == 'occupancy':
                collisions = cost_collision_self | cost_collision_objects | cost_collision_border
            else:
                collisions = cost_collision_self + cost_collision_objects + cost_collision_border

        return collisions

    def get_trajs_collision_and_free(
            self, trajs, return_indices=False, num_interpolation=0,
            **kwargs
    ):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:  # n_goals (or steps), batch of trajectories, length, dim
            N, B, H, D = trajs.shape
            trajs_new = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape
            trajs_new = trajs

        ###############################################################################################################
        # compute collisions on a finer interpolated trajectory
        if num_interpolation > 0:
            trajs_interpolated = interpolate_traj_via_points(trajs_new, num_interpolation=num_interpolation)
        else:
            trajs_interpolated = trajs_new
        # Set a low margin for collision checking, which means we allow trajectories to pass very close to objects.
        # While the optimized trajectory via points are not at a 0 margin from the object, the interpolated via points
        # might be. A 0 margin guarantees that we do not discard those trajectories, while ensuring they are not in
        # collision (margin < 0).
        trajs_waypoints_collisions = self.compute_collision(trajs_interpolated, margin=self.margin_for_waypoint_collision_checking)

        if trajs.ndim == 4:
            trajs_waypoints_collisions = einops.rearrange(trajs_waypoints_collisions, '(N B) H -> N B H', N=N, B=B)

        trajs_free_idxs = torch.argwhere(torch.logical_not(trajs_waypoints_collisions).all(dim=-1))
        trajs_coll_idxs = torch.argwhere(trajs_waypoints_collisions.any(dim=-1))

        ###############################################################################################################
        # Check that trajectories that are not in collision are inside the joint limits
        if trajs_free_idxs.nelement() == 0:
            pass
        else:
            if trajs.ndim == 4:
                trajs_free_tmp = trajs[trajs_free_idxs[:, 0], trajs_free_idxs[:, 1], ...]
            else:
                trajs_free_tmp = trajs[trajs_free_idxs.squeeze(), ...]

            trajs_free_tmp_position = self.robot.get_position(trajs_free_tmp)
            trajs_free_inside_joint_limits_idxs = torch.logical_and(
                trajs_free_tmp_position >= self.robot.q_min,
                trajs_free_tmp_position <= self.robot.q_max).all(dim=-1).all(dim=-1)
            trajs_free_inside_joint_limits_idxs = torch.atleast_1d(trajs_free_inside_joint_limits_idxs)
            trajs_free_idxs_try = trajs_free_idxs[torch.argwhere(trajs_free_inside_joint_limits_idxs).squeeze()]
            if trajs_free_idxs_try.nelement() == 0:
                trajs_coll_idxs = trajs_free_idxs.clone()
            else:
                trajs_coll_idxs_joint_limits = trajs_free_idxs[torch.argwhere(torch.logical_not(trajs_free_inside_joint_limits_idxs)).squeeze()]
                if trajs_coll_idxs_joint_limits.ndim == 1:
                    trajs_coll_idxs_joint_limits = trajs_coll_idxs_joint_limits[..., None]
                trajs_coll_idxs = torch.cat((trajs_coll_idxs, trajs_coll_idxs_joint_limits))
            trajs_free_idxs = trajs_free_idxs_try

        ###############################################################################################################
        # Return trajectories free and in collision
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
        _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.get_trajs_collision_and_free(trajs, return_indices=True, **kwargs)
        n_trajs_free = trajs_free_idxs.nelement()
        n_trajs_coll = trajs_coll_idxs.nelement()
        return n_trajs_free/(n_trajs_free + n_trajs_coll)

    def compute_collision_intensity_trajs(self, trajs, **kwargs):
        # Compute the fraction of waypoints that are in collision
        _, _, _, _, trajs_waypoints_collisions = self.get_trajs_collision_and_free(trajs, return_indices=True, **kwargs)
        return torch.count_nonzero(trajs_waypoints_collisions)/trajs_waypoints_collisions.nelement()

    def compute_success_free_trajs(self, trajs, **kwargs):
        # If at least one trajectory is collision free, then we consider success
        _, trajs_free = self.get_trajs_collision_and_free(trajs, **kwargs)
        if trajs_free is not None:
            if trajs_free.nelement() >= 1:
                return 1
        return 0



    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    # vizualization
    def render_robot_trajectories(self, fig=None, ax=None, render_planner=False, trajs=None, traj_best=None, **kwargs):
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        if render_planner:
            self.planner.render(ax)
        self.env.render(ax)
        if trajs is not None:
            _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.get_trajs_collision_and_free(trajs, return_indices=True)
            kwargs['colors'] = []
            for i in range(len(trajs_coll_idxs) + len(trajs_free_idxs)):
                kwargs['colors'].append(self.colors['collision'] if i in trajs_coll_idxs else self.colors['free'])
        self.robot.render_trajectories(ax, trajs=trajs, **kwargs)
        if traj_best is not None:
            kwargs['colors'] = ['blue']
            self.robot.render_trajectories(ax, trajs=traj_best.unsqueeze(0), **kwargs)

        return fig, ax

    def animate_robot_trajectories(
            self, trajs=None, start_state=None, goal_state=None,
            plot_trajs=False,
            n_frames=10,
            **kwargs
    ):
        if trajs is None:
            return

        assert trajs.ndim == 3
        B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_selection = trajs[:, idxs, :]

        fig, ax = create_fig_and_axes(dim=self.env.dim)
        def animate_fn(i):
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H-1}")
            if plot_trajs:
                self.render_robot_trajectories(
                    fig=fig, ax=ax, trajs=trajs, start_state=start_state, goal_state=goal_state, **kwargs
                )
            else:
                self.env.render(ax)

            # TODO - implement batched version
            qs = trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            for q in qs:
                self.robot.render(
                    ax, q=q,
                    color=self.colors_robot['collision'] if self.compute_collision(q, margin=0.0) else self.colors_robot['free'],
                    arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                    cmap=self.cmaps['collision'] if self.compute_collision(q, margin=0.0) else self.cmaps['free'],
                    **kwargs
                )

            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_opt_iters_robots(
            self, trajs=None, traj_best=None, start_state=None, goal_state=None,
            control_points=None,
            n_frames=10,
            **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]
        if control_points is None:
            # Assume the control points are the trajectory waypoints
            control_points_selection = trajs_selection
        else:
            control_points_selection = control_points[idxs]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S-1}")
            self.render_robot_trajectories(
                fig=fig, ax=ax, trajs=trajs_selection[i],
                control_points=control_points_selection[i],
                traj_best=traj_best if i == n_frames - 1 else None,
                start_state=start_state, goal_state=goal_state, **kwargs
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def plot_joint_space_state_trajectories(
            self,
            fig=None, axs=None,
            trajs=None,
            trajs_vel=None,
            trajs_acc=None,
            traj_best=None,
            pos_start_state=None, pos_goal_state=None,
            vel_start_state=None, vel_goal_state=None,
            acc_start_state=None, acc_goal_state=None,
            set_joint_limits=True,
            set_joint_vel_limits=True,
            set_joint_acc_limits=True,
            control_points=None,
            **kwargs
    ):
        if trajs is None:
            return
        trajs_np = to_numpy(trajs)

        assert trajs_np.ndim == 3
        B, H, D = trajs_np.shape

        # Separate trajectories in collision and free (not in collision)
        trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, _ = self.get_trajs_collision_and_free(trajs, return_indices=True, **kwargs)

        trajs_coll_pos_np = to_numpy([])
        trajs_coll_vel_np = to_numpy([])
        trajs_coll_acc_np = to_numpy([])
        if trajs_coll is not None:
            trajs_coll_pos_np = to_numpy(self.robot.get_position(trajs_coll))
            if trajs_vel is not None:
                trajs_coll_vel_np = to_numpy(trajs_vel[trajs_coll_idxs.squeeze()])
                if trajs_coll_vel_np.ndim == 2:
                    trajs_coll_vel_np = trajs_coll_vel_np[None, ...]
            else:
                trajs_coll_vel_np = to_numpy(self.robot.get_velocity(trajs_coll))
            if trajs_acc is not None:
                trajs_coll_acc_np = to_numpy(trajs_acc[trajs_coll_idxs.squeeze()])
                if trajs_coll_acc_np.ndim == 2:
                    trajs_coll_acc_np = trajs_coll_acc_np[None, ...]
            else:
                trajs_coll_acc_np = to_numpy(self.robot.get_acceleration(trajs_coll))

        trajs_free_pos_np = to_numpy([])
        trajs_free_vel_np = to_numpy([])
        trajs_free_acc_np = to_numpy([])
        if trajs_free is not None:
            trajs_free_pos_np = to_numpy(self.robot.get_position(trajs_free))
            if trajs_vel is not None:
                trajs_free_vel_np = to_numpy(trajs_vel[trajs_free_idxs.squeeze()])
                if trajs_free_vel_np.ndim == 2:
                    trajs_free_vel_np = trajs_free_vel_np[None, ...]
            else:
                trajs_free_vel_np = to_numpy(self.robot.get_velocity(trajs_free))
            if trajs_acc is not None:
                trajs_free_acc_np = to_numpy(trajs_acc[trajs_free_idxs.squeeze()])
                if trajs_free_acc_np.ndim == 2:
                    trajs_free_acc_np = trajs_free_acc_np[None, ...]
            else:
                trajs_free_acc_np = to_numpy(self.robot.get_acceleration(trajs_free))

        if pos_start_state is not None:
            pos_start_state = to_numpy(pos_start_state)
        if vel_start_state is not None:
            vel_start_state = to_numpy(vel_start_state)
        if acc_start_state is not None:
            acc_start_state = to_numpy(acc_start_state)
        if pos_goal_state is not None:
            pos_goal_state = to_numpy(pos_goal_state)
        if vel_goal_state is not None:
            vel_goal_state = to_numpy(vel_goal_state)
        if acc_goal_state is not None:
            acc_goal_state = to_numpy(acc_goal_state)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 3, squeeze=False)
        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[0, 2].set_title('Acceleration')
        axs[-1, 0].set_xlabel('Timesteps')
        axs[-1, 1].set_xlabel('Timesteps')
        axs[-1, 2].set_xlabel('Timesteps')
        timesteps = np.linspace(0, 1, H).reshape(1, -1)
        for i, ax in enumerate(axs):
            for trajs_filtered, color in zip([(trajs_coll_pos_np, trajs_coll_vel_np, trajs_coll_acc_np),
                                              (trajs_free_pos_np, trajs_free_vel_np, trajs_free_acc_np)],
                                             ['black', 'orange']):
                # Positions and velocities
                for j, trajs_filtered_ in enumerate(trajs_filtered):
                    if trajs_filtered_.size > 0:
                        timesteps_ = np.repeat(timesteps, trajs_filtered_.shape[0], axis=0)
                        try:
                            plot_multiline(ax[j], timesteps_, trajs_filtered_[..., i], color=color, **kwargs)
                        except:
                            pass

            if traj_best is not None:
                traj_best_pos = self.robot.get_position(traj_best)
                traj_best_vel = self.robot.get_velocity(traj_best)
                traj_best_acc = self.robot.get_acceleration(traj_best)
                traj_best_pos_np = to_numpy(traj_best_pos)
                traj_best_vel_np = to_numpy(traj_best_vel)
                traj_best_acc_np = to_numpy(traj_best_acc)
                plot_multiline(ax[0], timesteps, traj_best_pos_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[1], timesteps, traj_best_vel_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[2], timesteps, traj_best_acc_np[..., i].reshape(1, -1), color='blue', **kwargs)

            # Start and goal
            if pos_start_state is not None:
                ax[0].scatter(0, pos_start_state[i], color='green')
            if vel_start_state is not None:
                ax[1].scatter(0, vel_start_state[i], color='green')
            if acc_start_state is not None:
                ax[2].scatter(0, acc_start_state[i], color='green')
            if pos_goal_state is not None:
                ax[0].scatter(1, pos_goal_state[i], color='purple')
            if vel_goal_state is not None:
                ax[1].scatter(1, vel_goal_state[i], color='purple')
            if acc_goal_state is not None:
                ax[2].scatter(1, acc_goal_state[i], color='purple')
            # Y label
            ax[0].set_ylabel(f'q_{i}')
            # Set limits
            if set_joint_limits:
                ax[0].plot([0, 1], [self.robot.q_min_np[i], self.robot.q_min_np[i]], color='k', linestyle='--')
                ax[0].plot([0, 1], [self.robot.q_max_np[i], self.robot.q_max_np[i]], color='k', linestyle='--')
                # ax[0].set_ylim(self.robot.q_min_np[i], self.robot.q_max_np[i])
            if set_joint_vel_limits and self.robot.dq_max_np is not None:
                ax[1].plot([0, 1], [self.robot.dq_max_np[i], self.robot.dq_max_np[i]], color='k', linestyle='--')
                ax[1].plot([0, 1], [-self.robot.dq_max_np[i], -self.robot.dq_max_np[i]], color='k', linestyle='--')
                # ax[1].set_ylim(-self.robot.dq_max_np[i], self.robot.dq_max_np[i])
            if set_joint_acc_limits and self.robot.ddq_max_np is not None:
                ax[2].plot([0, 1], [self.robot.ddq_max_np[i], self.robot.ddq_max_np[i]], color='k', linestyle='--')
                ax[2].plot([0, 1], [-self.robot.ddq_max_np[i], -self.robot.ddq_max_np[i]], color='k', linestyle='--')
                # ax[2].set_ylim(-self.robot.ddq_max_np[i], self.robot.ddq_max_np[i])

        # plot control points
        if control_points is not None:
            control_points_np = to_numpy(control_points)
            control_points_timesteps = np.linspace(0, 1, control_points_np.shape[1])
            for control_points_np_one in control_points_np:
                for i, ax in enumerate(axs):
                    ax[0].scatter(control_points_timesteps, control_points_np_one[:, i], color='red', s=2 ** 2, zorder=10)

        return fig, axs

    def animate_opt_iters_joint_space_state(
            self, trajs=None, traj_best=None, n_frames=10,
            trajs_vel=None, trajs_acc=None,
            **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]
        trajs_vel_selection = None
        if trajs_vel is not None:
            trajs_vel_selection = trajs_vel[idxs]
        trajs_acc_selection = None
        if trajs_acc is not None:
            trajs_acc_selection = trajs_acc[idxs]

        fig, axs = self.plot_joint_space_state_trajectories(
            trajs=trajs_selection[0],
            trajs_vel=trajs_vel_selection[0] if trajs_vel is not None else None,
            trajs_acc=trajs_acc_selection[0] if trajs_acc is not None else None,
            **kwargs)

        def animate_fn(i):
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S-1}")
            self.plot_joint_space_state_trajectories(
                fig=fig, axs=axs,
                trajs=trajs_selection[i],
                trajs_vel=trajs_vel_selection[i] if trajs_vel is not None else None,
                trajs_acc=trajs_acc_selection[i] if trajs_acc is not None else None,
                **kwargs
            )
            if i == n_frames - 1 and traj_best is not None:
                self.plot_joint_space_state_trajectories(
                    fig=fig, axs=axs,
                    trajs=trajs_selection[i],
                    traj_best=traj_best,
                    trajs_vel=trajs_vel_selection[i] if trajs_vel is not None else None,
                    trajs_acc=trajs_acc_selection[i] if trajs_acc is not None else None,
                    **kwargs
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)


