import einops
import torch
from matplotlib import pyplot as plt

from mp_baselines.planners.utils import extend_path, to_numpy
from robot_envs.base_envs.env_base import EnvBase
from torch_planning_objectives.fields.distance_fields import EmbodimentDistanceField
from torch_planning_objectives.fields.primitive_distance_fields import Sphere


class PointMassEnvBase(EnvBase):

    def __init__(self,
                 name='pointmass_env',
                 q_min=(-1, -1),
                 q_max=(1, 1),
                 obst_primitives_l=None,
                 task_space_bounds=((-1., 1.), (-1., 1.), (-1., 1.)),
                 obstacle_buffer=0.01,
                 self_buffer=0.0005,
                 tensor_args=None
                 ):
        ################################################################################################
        # Robot Point Mass
        q_min = torch.tensor(q_min)
        q_max = torch.tensor(q_max)
        q_n_dofs = len(q_min)

        super().__init__(name=name, q_n_dofs=q_n_dofs, q_min=q_min, q_max=q_max, tensor_args=tensor_args)

        ################################################################################################
        # Task space dimensions
        self.task_space_bounds = task_space_bounds
        self.task_space_bounds_min = torch.Tensor([task_space_bounds[0][0],
                                                   task_space_bounds[1][0],
                                                   task_space_bounds[2][0]]).to(**tensor_args)
        self.task_space_bounds_max = torch.Tensor([task_space_bounds[0][1],
                                                   task_space_bounds[1][1],
                                                   task_space_bounds[2][1]]).to(**tensor_args)

        ################################################################################################
        # Obstacles
        self.obst_primitives_l = obst_primitives_l

        ################################################################################################
        # Collisions
        self.obstacle_buffer = obstacle_buffer
        self.self_buffer = self_buffer
        self.df_collision_self_and_obstacles = EmbodimentDistanceField(
            self_margin=self_buffer, obst_margin=obstacle_buffer,
            num_interpolate=4, link_interpolate_range=[2, 7]
        )

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

        link_tensor = q

        # reshape to batch, trajectory, link poses
        link_tensor = einops.rearrange(link_tensor, '(b h) d -> b h 1 d', b=b, h=h)
        link_pos = link_tensor

        ########################
        # No need to compute self collisions in a point mass
        # Obstacle collision
        cost_collision_obstacle = self.df_collision_self_and_obstacles.compute_obstacle_cost(
            link_pos, df_list=self.obst_primitives_l, field_type=field_type)

        cost_collision = cost_collision_obstacle

        return cost_collision

    def _compute_collision_cost(self, q, field_type='sdf', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def _compute_collision(self, q, field_type='occupancy', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def get_rrt_params(self):
        # RRT planner parameters
        params = dict(
            env=self,
            n_iters=10000,
            max_best_cost_iters=1000,
            cost_eps=1e-2,
            step_size=0.01,
            n_radius=0.1,
            n_knn=10,
            max_time=60.,
            goal_prob=0.1,
            tensor_args=self.tensor_args
        )
        return params

    def get_sgpmp_params(self):
        # SGPMP planner parameters
        params = dict(
            dt=0.02,
            n_dof=self.q_n_dofs,
            temperature=1.,
            step_size=1.,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.4,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            sigma_gp_sample=0.2,
            tensor_args=self.tensor_args,
        )
        return params

    def render(self, ax=None):
        # plot obstacles
        for obst_primitive in self.obst_primitives_l:
            obst_primitive.draw(ax)

        ax.set_xlim(*self.task_space_bounds[0])
        ax.set_ylim(*self.task_space_bounds[1])



if __name__ == "__main__":
    tensor_args = dict(device='cpu', dtype=torch.float32)

    obst_primitives_l = [
        Sphere(
            [[0.3, 0.3],
             [0.6, 0.6]],
            [0.1, 0.2],
            tensor_args=tensor_args
        )
    ]

    env = PointMassEnvBase(
        obst_primitives_l=obst_primitives_l,
        tensor_args=tensor_args
    )

    q_start = env.random_coll_free_q(max_samples=1000)
    q_goal = env.random_coll_free_q(max_samples=1000)
    path = extend_path(env.distance_q, q_start, q_goal, max_step=0.1, max_dist=torch.inf, tensor_args=tensor_args)
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax)
    # plot path
    if path is not None:
        traj = to_numpy(path)
        traj_pos = env.get_q_position(traj)
        ax.plot(traj_pos[:, 0], traj_pos[:, 1])
        ax.scatter(traj_pos[0][0], traj_pos[0][1], color='green', s=50)
        ax.scatter(traj_pos[-1][0], traj_pos[-1][1], color='red', s=50)
    plt.show()


