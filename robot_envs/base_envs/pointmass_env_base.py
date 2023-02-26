import einops
import torch
from matplotlib import pyplot as plt

from mp_baselines.planners.utils import extend_path, to_numpy
from robot_envs.base_envs.env_base import EnvBase
from robot_envs.base_envs.utils import plot_trajectories
from torch_planning_objectives.fields.distance_fields import EmbodimentDistanceField, BorderDistanceField
from torch_planning_objectives.fields.primitive_distance_fields import SphereField


class PointMassEnvBase(EnvBase):

    def __init__(self,
                 name='pointmass_env',
                 q_min=(-1, -1),
                 q_max=(1, 1),
                 obst_primitives_l=None,
                 work_space_dim=2,
                 work_space_bounds=((-1., 1.), (-1., 1.), (-1., 1.)),
                 obstacle_buffer=0.01,
                 self_buffer=0.0005,
                 tensor_args=None
                 ):
        ################################################################################################
        # Robot Point Mass
        q_min = torch.tensor(q_min)
        q_max = torch.tensor(q_max)
        q_n_dofs = len(q_min)

        super().__init__(name=name, q_n_dofs=q_n_dofs, q_min=q_min, q_max=q_max,
                         work_space_dim=work_space_dim, tensor_args=tensor_args)

        ################################################################################################
        # Task space dimensions
        self.work_space_bounds = work_space_bounds
        if self.work_space_dim == 3:
            self.work_space_bounds_min = torch.Tensor([work_space_bounds[0][0],
                                                       work_space_bounds[1][0],
                                                       work_space_bounds[2][0]]).to(**tensor_args)
            self.work_space_bounds_max = torch.Tensor([work_space_bounds[0][1],
                                                       work_space_bounds[1][1],
                                                       work_space_bounds[2][1]]).to(**tensor_args)
        elif self.work_space_dim == 2:
            self.work_space_bounds_min = torch.Tensor([work_space_bounds[0][0],
                                                       work_space_bounds[1][0]]).to(**tensor_args)
            self.work_space_bounds_max = torch.Tensor([work_space_bounds[0][1],
                                                       work_space_bounds[1][1]]).to(**tensor_args)
        else:
            raise NotImplementedError

        ################################################################################################
        # Obstacles
        self.obst_primitives_l = obst_primitives_l

        ################################################################################################
        # Collisions
        self.obstacle_buffer = obstacle_buffer
        self.self_buffer = self_buffer
        self.df_collision_self_and_obstacles = EmbodimentDistanceField(
            field_type='occupancy',
            self_margin=self_buffer, obst_margin=obstacle_buffer,
            tensor_args=tensor_args
        )

        self.df_collision_border = BorderDistanceField(
            work_space_bounds_min=self.work_space_bounds_min,
            work_space_bounds_max=self.work_space_bounds_max,
            field_type='occupancy',
            obst_margin=0.,
            tensor_args=tensor_args
        )

        ################################################################################################
        # Guides diffusion
        self.guide_scale_collision_avoidance = 3 * 1e-2
        self.guide_scale_smoothness_finite_diff_velocity = 1 * 1e-1
        self.guide_scale_gp_prior = 5 * 1e-3

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
            link_pos, df_list=self.obst_primitives_l, field_type=field_type, **kwargs)

        # Border collision
        cost_collision_border = self.df_collision_border.compute_cost(
            link_pos, field_type=field_type)

        if field_type == 'occupancy':
            cost_collision = cost_collision_obstacle | cost_collision_border
        else:
            cost_collision = cost_collision_obstacle + cost_collision_border

        return cost_collision

    def _compute_collision_cost(self, q, field_type='sdf', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def _compute_collision(self, q, field_type='occupancy', **kwargs):
        return self.compute_cost_collision_internal(q, field_type=field_type, **kwargs)

    def get_rrt_params(self):
        # RRT planner parameters
        params = dict(
            env=self,
            n_iters=50000,
            max_best_cost_iters=1000,
            cost_eps=1e-2,
            step_size=0.01,
            n_radius=0.3,
            n_knn=10,
            max_time=60.,
            goal_prob=0.2,
            n_pre_samples=50000,
            tensor_args=self.tensor_args
        )
        return params

    def render(self, ax=None):
        # plot obstacles
        for obst_primitive in self.obst_primitives_l:
            obst_primitive.draw(ax)

        ax.set_xlim(*self.work_space_bounds[0])
        ax.set_ylim(*self.work_space_bounds[1])
        if self.work_space_dim == 3:
            ax.set_zlim(*self.work_space_bounds[2])
            ax.view_init(azim=0, elev=90)

    def render_trajectories(self, ax=None, trajs=None, **kwargs):
        plot_trajectories(ax, trajs, **kwargs)



if __name__ == "__main__":
    tensor_args = dict(device='cpu', dtype=torch.float32)

    obst_primitives_l = [
        SphereField(
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


