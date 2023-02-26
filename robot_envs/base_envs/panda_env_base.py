import os
import time
from math import ceil

import time
from math import ceil

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt

from mp_baselines.planners.utils import extend_path, to_numpy
from robot_envs.base_envs.env_base import EnvBase
from robot_envs.pybullet.objects import Panda
from robot_envs.pybullet.panda import PandaEnv
from robot_envs.pybullet.panda_bullet_base import PandaEnvPyBulletBase
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_planning_objectives.fields.collision_bodies import PandaSphereDistanceField
from torch_planning_objectives.fields.distance_fields import EmbodimentDistanceField, BorderDistanceField
from torch_planning_objectives.fields.occupancy_map.map_generator import build_obstacle_map
from torch_planning_objectives.fields.primitive_distance_fields import SphereField, BoxField, InfiniteCylinderField


class PandaEnvBase(EnvBase):

    def __init__(self,
                 name='panda_simple_env',
                 obst_primitives_l=None,
                 work_space_bounds=((-1.25, 1.25), (-1.25, 1.25), (0, 1.5)),
                 obstacle_buffer=0.01,
                 self_buffer=0.0005,
                 compute_robot_collision_from_occupancy_grid=False,
                 tensor_args=None
                 ):
        ################################################################################################
        # Robot pybullet
        self.panda_robot = Panda()
        q_min = torch.tensor(self.panda_robot.jl_lower)
        q_max = torch.tensor(self.panda_robot.jl_upper)
        q_n_dofs = len(q_min)

        super().__init__(name=name, q_n_dofs=q_n_dofs, q_min=q_min, q_max=q_max,
                         work_space_dim=3, tensor_args=tensor_args)

        # Physics Environment
        self.panda_bullet_env = PandaEnvPyBulletBase(
            obst_primitives_l=obst_primitives_l,
            render=True
        )

        ################################################################################################
        # Task space dimensions
        self.work_space_bounds = work_space_bounds
        self.work_space_bounds_min = torch.Tensor([work_space_bounds[0][0],
                                                   work_space_bounds[1][0],
                                                   work_space_bounds[2][0]]).to(**tensor_args)
        self.work_space_bounds_max = torch.Tensor([work_space_bounds[0][1],
                                                   work_space_bounds[1][1],
                                                   work_space_bounds[2][1]]).to(**tensor_args)

        self.floor_min = torch.Tensor([0.]).to(**tensor_args)  # floor z position

        ################################################################################################
        # Obstacles
        self.obst_primitives_l = obst_primitives_l

        # optionally setup the obstacle map to use with the robot represented with spheres
        self.compute_robot_collision_from_occupancy_grid = compute_robot_collision_from_occupancy_grid
        self.obstacle_map = None
        if self.compute_robot_collision_from_occupancy_grid:
            self.panda_collision = PandaSphereDistanceField(tensor_args=tensor_args)
            self.panda_collision.build_batch_features(batch_dim=[1000, ], clone_objs=True)
            self.setup_obstacle_map()

        ################################################################################################
        # Collisions
        # Robot torch
        self.diff_panda = DifferentiableFrankaPanda(gripper=False, device=self.tensor_args['device'])
        self.link_names_for_collision_checking = ['panda_link1', 'panda_link3', 'panda_link4', 'panda_link5',
                                                  'panda_link7', 'panda_link8', 'ee_link']
        self.link_name_ee = 'ee_link'

        # Robot collision model
        self.obstacle_buffer = obstacle_buffer
        self.self_buffer = self_buffer
        self.df_collision_self_and_obstacles = EmbodimentDistanceField(
            self_margin=self_buffer, obst_margin=obstacle_buffer,
            field_type='occupancy',
            num_interpolate=4, link_interpolate_range=[2, 7]
        )

        self.df_collision_border = BorderDistanceField(
            work_space_bounds_min=self.work_space_bounds_min,
            work_space_bounds_max=self.work_space_bounds_max,
            field_type='occupancy',
            obst_margin=0.,
            tensor_args=tensor_args
        )

        self.df_collision_floor = None
        # self.floor_collision = FloorDistanceField(tensor_args=tensor_args)

        ################################################################################################
        # Guides diffusion
        self.guide_scale_collision_avoidance = 5 * 1e-2
        self.guide_scale_smoothness_finite_diff_velocity = 1e-1
        self.guide_scale_gp_prior = 5 * 1e-3
        self.guide_scale_se3_orientation_goal = 1 * 1e-2

        # Guides CVAE
        self.guide_scale_collision_avoidance_cvae = 5 * 1e-2
        self.guide_scale_smoothness_finite_diff_velocity_cvae = 1e-1
        self.guide_scale_gp_prior_cvae = 5 * 1e-3
        self.guide_scale_se3_orientation_goal_cvae = 1 * 1e-2


    def setup_obstacle_map(self):
        map_dim = [ceil((dim_bound[1] - dim_bound[0])/2.)*2 for dim_bound in self.work_space_bounds]

        obst_params = dict(
            map_dim=map_dim,
            obst_list=self.obst_primitives_l,
            cell_size=max(map_dim)/20,
            map_type='direct',
            tensor_args=tensor_args,
        )
        obst_map = build_obstacle_map(**obst_params)

        self.obstacle_map = obst_map

    def compute_cost_collision_internal(self, q, field_type='occupancy', **kwargs):
        b = 1
        h = 1
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(0)
        elif q.ndim == 3:
            b = q.shape[0]
            h = q.shape[1]
        elif q.ndim > 3:
            raise NotImplementedError

        # batch, trajectory length, q dimension
        q = einops.rearrange(q, 'b h d -> (b h) d')

        # link_tensor = self.diff_panda.compute_forward_kinematics_all_links(q)
        link_tensor = self.diff_panda.compute_forward_kinematics_link_list(q, link_list=self.link_names_for_collision_checking)

        # reshape to batch, trajectory, link poses
        link_tensor = einops.rearrange(link_tensor, '(b h) links d1 d2 -> b h links d1 d2', b=b, h=h)
        link_pos = link_pos_from_link_tensor(link_tensor)

        if self.compute_robot_collision_from_occupancy_grid:
            cost_collision = self.collision_robot_occupancy_grid(q, batch_dim=b)
        else:
            ########################
            # Self collision and Obstacle collision
            cost_collision_self, cost_collision_obstacle = self.df_collision_self_and_obstacles.compute_cost(
                link_pos, df_list=self.obst_primitives_l, field_type=field_type
            )

            ########################
            # Border collision
            cost_collision_border = self.df_collision_border.compute_cost(
                link_pos, field_type=field_type)

            ########################
            # Floor collision
            # collision_floor = self.df_collision_floor.compute_cost()

            if field_type == 'occupancy':
                cost_collision = cost_collision_self | cost_collision_obstacle | cost_collision_border
            else:
                cost_collision = cost_collision_self + cost_collision_obstacle + cost_collision_border

        return cost_collision

    def _compute_collision_cost(self, q, field_type='sdf', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def _compute_collision(self, q, field_type='occupancy', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def collision_robot_occupancy_grid(self, q, batch_dim=1):
        # Computes the collisions between the robot spheres and the occupancy grid
        links_dict = self.diff_panda.compute_forward_kinematics_all_links(q, return_dict=True)
        # TODO: build_batch_features only when batch dimension changes
        self.panda_collision.build_batch_features(batch_dim=[batch_dim, ], clone_objs=True)
        self.panda_collision.update_batch_robot_collision_objs(links_dict)
        link_tensor_spheres = self.panda_collision.get_batch_robot_link_spheres()
        link_tensor_spheres_flat = torch.cat(link_tensor_spheres, dim=-2)
        link_tensor_spheres_centers = link_tensor_spheres_flat[..., :3]
        link_tensor_spheres_radii = link_tensor_spheres_flat[..., 3]
        distance_grid_points_to_spheres_centers = self.obstacle_map.compute_distances(link_tensor_spheres_centers)
        distance_grid_points_to_spheres_centers_min = torch.min(distance_grid_points_to_spheres_centers, dim=-1)[0]
        obstacle_collision = torch.any(distance_grid_points_to_spheres_centers_min < link_tensor_spheres_radii, dim=-1, keepdim=True)
        return obstacle_collision

    def get_rrt_params(self):
        # RRT planner parameters
        params = dict(
            env=self,
            n_iters=50000,
            max_best_cost_iters=1000,
            cost_eps=1e-2,
            step_size=np.pi/16,
            n_radius=np.pi/2,
            n_knn=10,
            max_time=30.,
            goal_prob=0.2,
            n_pre_samples=50000,
            tensor_args=self.tensor_args
        )
        return params

    def render(self, ax=None):
        # plot obstacles
        for obst_primitive in self.obst_primitives_l:
            obst_primitive.draw(ax)

        ax.view_init(azim=0, elev=90)
        ax.set_xlim3d(*self.work_space_bounds[0])
        ax.set_ylim3d(*self.work_space_bounds[1])
        ax.set_zlim3d(*self.work_space_bounds[2])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def render_trajectories(self, ax=None, traj_l=None, color='orange', plot_only_one=False, **kwargs):
        # plot path
        if traj_l is not None:
            for traj in traj_l:
                for t in range(traj.shape[0] - 1):
                    skeleton = get_skeleton_from_model(self.diff_panda, traj[t], self.diff_panda.get_link_names())
                    skeleton.draw_skeleton(ax=ax, color=color)
                start_skeleton = get_skeleton_from_model(self.diff_panda, traj[0], self.diff_panda.get_link_names())
                start_skeleton.draw_skeleton(ax=ax, color='green')
                goal_skeleton = get_skeleton_from_model(self.diff_panda, traj[-1], self.diff_panda.get_link_names())
                goal_skeleton.draw_skeleton(ax=ax, color='red')
                if plot_only_one:
                    break

    def render_physics(self, traj=None, path_video_file=None, **kwargs):
        if traj is not None:
            traj = to_numpy(traj)
            self.panda_bullet_env.reset(robot_q=traj[0], **kwargs)
            if path_video_file is not None:
                id = self.panda_bullet_env.client_id.startStateLogging(
                        self.panda_bullet_env.client_id.STATE_LOGGING_VIDEO_MP4,
                        path_video_file
                    )
            for t in range(traj.shape[0] - 1):
                action = np.concatenate((traj[t], np.zeros(5)))
                self.panda_bullet_env.step(action)

            if path_video_file is not None:
                self.panda_bullet_env.client_id.stopStateLogging(id)


if __name__ == "__main__":
    tensor_args = dict(device='cpu', dtype=torch.float32)

    obst_primitives_l = [
        SphereField(
            [[0.5, 0.5, 0.5],
             [-0.8, -0.8, 0.8]],
            [0.4, 0.2],
            tensor_args=tensor_args
        ),
        BoxField(
            [[-0.5, 0.5, 0.5],
             [0., 0.2, 0.5],
             ],
            [[0.6, 0.5, 0.4],
             [0.2, 0.2, 0.2]
             ],
            tensor_args=tensor_args
        )
    ]

    env = PandaEnvBase(
        obst_primitives_l=obst_primitives_l,
        tensor_args=tensor_args
    )

    q_start = env.random_coll_free_q(max_samples=1000)
    q_goal = env.random_coll_free_q(max_samples=1000)
    path = extend_path(env.distance_q, q_start, q_goal, max_step=np.pi/10, max_dist=torch.inf, tensor_args=tensor_args)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    env.render(ax)
    env.render_trajectories(ax, [path])
    plt.show()

    target = env.diff_panda.compute_forward_kinematics_link_list(path[-1].unsqueeze(0), link_list=['ee_link']).squeeze()
    target = to_numpy(target)
    env.render_physics(path, target_EE=target)

