import sys
import time
from math import ceil

import numpy as np

import torch
from matplotlib import pyplot as plt
import einops

from mp_baselines.planners.utils import extend_path, to_numpy
from robot_envs.base_envs.env_base import EnvBase
from robot_envs.base_envs.obstacle_map_env import ObstacleMapEnv
from robot_envs.pybullet.objects import Panda

from robot_envs.pybullet.panda import PandaEnv
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_planning_objectives.fields.collision_bodies import PandaSphereDistanceField
from torch_planning_objectives.fields.distance_fields import LinkSelfDistanceField, LinkDistanceField, \
    FloorDistanceField, BorderDistanceField
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map
from torch_planning_objectives.fields.occupancy_map.obst_map import ObstacleSphere


class PandaEnv7D(EnvBase):

    def __init__(self,
                 name='panda_simple_env',
                 obstacle_spheres=None,
                 task_space_bounds=((-1.25, 1.25), (-1.25, 1.25), (0, 1.5)),
                 obstacle_buffer=0.08,
                 self_buffer=0.05,
                 compute_robot_collision_from_occupancy_grid=False,
                 tensor_args=None
                 ):
        ################################################################################################
        # Robot pybullet
        self.panda_robot = Panda()
        q_min = torch.tensor(self.panda_robot.jl_lower)
        q_max = torch.tensor(self.panda_robot.jl_upper)
        q_n_dofs = len(q_min)

        super().__init__(name=name, q_n_dofs=q_n_dofs, q_min=q_min, q_max=q_max, tensor_args=tensor_args)

        # Physics Environment
        self.panda_bullet_env = PandaEnv(render=True)

        ################################################################################################
        # Task space dimensions
        self.task_space_bounds = task_space_bounds
        self.task_space_bounds_min = torch.Tensor([task_space_bounds[0][0],
                                                   task_space_bounds[1][0],
                                                   task_space_bounds[2][0]]).to(**tensor_args)
        self.task_space_bounds_max = torch.Tensor([task_space_bounds[0][1],
                                                   task_space_bounds[1][1],
                                                   task_space_bounds[2][1]]).to(**tensor_args)

        self.floor_min = torch.Tensor([0.]).to(**tensor_args)  # floor z position

        ################################################################################################
        # Obstacles
        self.obstacle_spheres = None
        self.setup_obstacles(obstacle_spheres=obstacle_spheres)

        self.obstacle_buffer = obstacle_buffer

        self.self_buffer = self_buffer

        # optionally setup the obstacle map to use with the robot represented with spheres
        self.compute_robot_collision_from_occupancy_grid = compute_robot_collision_from_occupancy_grid
        self.obstacle_map = None
        if self.compute_robot_collision_from_occupancy_grid:
            self.setup_obstacle_map(obstacle_spheres=obstacle_spheres)

        ################################################################################################
        # Collisions
        # Robot torch
        self.diff_panda = DifferentiableFrankaPanda(gripper=False, device=self.tensor_args['device'])

        # Robot collision model
        self.panda_collision = PandaSphereDistanceField(tensor_args=tensor_args)
        self.panda_collision.build_batch_features(batch_dim=[1000, ], clone_objs=True)

        self.self_collision = LinkSelfDistanceField(tensor_args=tensor_args)

        self.floor_collision = FloorDistanceField(tensor_args=tensor_args)

        self.obstacle_collision = LinkDistanceField(tensor_args=tensor_args)

        self.border_collision = BorderDistanceField(tensor_args=tensor_args)

    def setup_obstacles(self, obstacle_spheres=None):
        self.obstacle_spheres = torch.zeros(len(obstacle_spheres), 4).to(**self.tensor_args)
        for i, sphere in enumerate(obstacle_spheres):
            self.obstacle_spheres[i, :] = torch.tensor([*sphere]).to(**self.tensor_args)

    def setup_obstacle_map(self, obstacle_spheres=None):
        map_dim = [ceil((dim_bound[1] - dim_bound[0])/2.)*2 for dim_bound in self.task_space_bounds]
        obst_list = [ObstacleSphere(sphere[:3], sphere[3]) for sphere in obstacle_spheres]
        obst_params = dict(
            map_dim=map_dim,
            obst_list=obst_list,
            cell_size=max(map_dim)/20,
            map_type='direct',
            tensor_args=self.tensor_args,
        )
        obst_map, obst_list = generate_obstacle_map(**obst_params)
        self.obstacle_map = obst_map

    def compute_collision(self, q, **kwargs):
        b = 1
        h = 1
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(0)
        elif q.nim == 3:
            b = q.shape[0], h = q.shape[1]
        elif q.ndim > 3:
            raise NotImplementedError

        # batch, trajectory length, q dimension
        q = einops.rearrange(q, 'b h d -> (b h) d')

        link_tensor = self.diff_panda.compute_forward_kinematics_all_links(q)

        # reshape to batch, trajectory, link poses
        link_tensor = einops.rearrange(link_tensor, '(b h) links d1 d2 -> b h links d1 d2', b=b, h=h)

        if self.compute_robot_collision_from_occupancy_grid:
            collisions = self.collision_robot_occupancy_grid(q, batch_dim=b)
        else:
            ########################
            # Self collision
            self_collision = self.self_collision.compute_collision(link_tensor, buffer=self.self_buffer)

            ########################
            # Obstacle collision
            obstacle_collision = self.obstacle_collision.compute_collision(link_tensor, obstacle_spheres=self.obstacle_spheres, buffer=self.obstacle_buffer)

            ########################
            # Floor collision
            floor_collision = self.floor_collision.compute_collision(link_tensor, self.floor_min)

            ########################
            # Border collision
            border_collision = self.border_collision.compute_collision(link_tensor, self.task_space_bounds_min, self.task_space_bounds_max)

            collisions = self_collision | obstacle_collision | floor_collision | border_collision

        return collisions

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
        obstacle_collision = torch.any(distance_grid_points_to_spheres_centers_min < link_tensor_spheres_radii,
                                       dim=-1, keepdim=True)
        return obstacle_collision

    def draw_sphere(self, ax, sphere):
        sphere = to_numpy(sphere)
        center = sphere[:3]
        radius = sphere[3]
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = radius * (np.cos(u) * np.sin(v))
        y = radius * (np.sin(u) * np.sin(v))
        z = radius * np.cos(v)
        ax.plot_surface(x + center[0], y + center[1], z + center[2], cmap='gray', alpha=0.75)

    def render(self, traj=None, ax=None):
        # plot obstacles
        for sphere in self.obstacle_spheres:
            self.draw_sphere(ax, sphere)

        # plot path
        if traj is not None:
            traj = torch.Tensor(traj)
            for t in range(traj.shape[0] - 1):
                skeleton = get_skeleton_from_model(self.diff_panda, traj[t], self.diff_panda.get_link_names())
                skeleton.draw_skeleton()
            skeleton = get_skeleton_from_model(self.diff_panda, traj[-1], self.diff_panda.get_link_names())
            skeleton.draw_skeleton(color='g')
            start_skeleton = get_skeleton_from_model(self.diff_panda, traj[0], self.diff_panda.get_link_names())
            start_skeleton.draw_skeleton(color='r')

        ax.set_xlim(*self.task_space_bounds[0])
        ax.set_ylim(*self.task_space_bounds[1])
        ax.set_zlim(*self.task_space_bounds[2])

    def render_physics(self, traj=None):
        if traj is not None:
            self.panda_bullet_env.reset()
            for t in range(traj.shape[0] - 1):
                self.panda_bullet_env.step(to_numpy(torch.cat((traj[t], torch.tensor([0, 0, 0, 0, 0])))))
                time.sleep(0.5)


if __name__ == "__main__":
    tensor_args = dict(device='cpu', dtype=torch.float32)

    spheres = [
         (0.5, 0.5, 0.5, 0.4)
    ]

    env = PandaEnv7D(
        obstacle_spheres=spheres,
        tensor_args=tensor_args
    )

    q_start = env.random_coll_free_q()
    q_goal = env.random_coll_free_q()
    path = extend_path(env.distance_q, q_start, q_goal, max_step=np.pi/10, max_dist=torch.inf, tensor_args=tensor_args)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    env.render(path, ax)
    plt.show()

    env.render_physics(path)

