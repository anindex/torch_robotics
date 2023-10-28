import abc
import itertools
from abc import ABC
from math import ceil

import einops
import torch

from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionSelfField
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from torch_robotics.trajectory.utils import finite_difference_vector

import torchkin as kin


class RobotBase(ABC):

    def __init__(
            self,
            name='RobotBase',
            q_limits=None,
            grasped_object=None,
            margin_for_grasped_object_collision_checking=0.001,
            link_names_for_object_collision_checking=None,
            link_margins_for_object_collision_checking=None,
            link_idxs_for_object_collision_checking=None,
            link_names_for_self_collision_checking=None,
            link_names_pairs_for_self_collision_checking=None,
            link_idxs_for_self_collision_checking=None,
            self_collision_margin_robot=0.001,
            link_names_for_self_collision_checking_with_grasped_object=None,
            self_collision_margin_grasped_object=0.05,
            num_interpolated_points_for_self_collision_checking=50,
            num_interpolated_points_for_object_collision_checking=50,
            dt=1.0,  # time interval to compute velocities and accelerations from positions via finite difference
            use_collision_spheres=False,
            robot_urdf_path=None,
            robot_urdf_path_ompl=None,
            link_names_torchkin=None,
            tensor_args=None,
            **kwargs
    ):
        self.name = name
        self.tensor_args = tensor_args

        self.dt = dt

        self.robot_urdf_path_ompl = robot_urdf_path_ompl

        ################################################################################################
        # torchkin robot
        self.robot_urdf_path = robot_urdf_path

        self.robot_torchkin = kin.Robot.from_urdf_file(robot_urdf_path, **tensor_args)
        # Print robot name, number of links and degrees of freedom
        print(f"{self.robot_torchkin.name} has {len(self.robot_torchkin.get_links())} links and {self.robot_torchkin.dof} degrees of freedom.\n")

        # Print joint id and name
        # for id, name in enumerate(self.robot_torchkin.joint_map):
        #     # A joint is not fixed if and only if id < robot.dof
        #     print(f"joint {id}: {name} is {'not fixed' if id < self.robot_torchkin.dof else 'fixed'}")
        # print("\n")

        # Print link id and name
        # for link in self.robot_torchkin.get_links():
        #     print(f"link {link.id}: {link.name}")

        self.link_names_torchkin = link_names_torchkin
        fk, jfk_b, jfk_s = kin.get_forward_kinematics_fns(robot=self.robot_torchkin, link_names=self.link_names_torchkin)
        self.robot_torchkin_fk = fk
        self.robot_torchkin_jfk_b = jfk_b
        self.robot_torchkin_jfk_s = jfk_s

        ################################################################################################
        # Configuration space
        assert q_limits is not None, "q_limits cannot be None"
        self.q_limits = q_limits
        self.q_min = q_limits[0]
        self.q_max = q_limits[1]
        self.q_min_np = to_numpy(self.q_min)
        self.q_max_np = to_numpy(self.q_max)
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)
        self.q_dim = len(self.q_min)

        ################################################################################################
        # Grasped object
        self.grasped_object = grasped_object
        self.margin_for_grasped_object_collision_checking = margin_for_grasped_object_collision_checking

        ################################################################################################
        # Objects collision field
        self.use_collision_spheres = use_collision_spheres

        assert num_interpolated_points_for_object_collision_checking >= len(link_names_for_object_collision_checking)
        if num_interpolated_points_for_object_collision_checking % len(link_names_for_object_collision_checking) != 0:
            self.points_per_link_object_collision_checking = ceil(num_interpolated_points_for_object_collision_checking / len(link_names_for_object_collision_checking))
            num_interpolated_points_for_object_collision_checking = self.points_per_link_object_collision_checking * len(link_names_for_object_collision_checking)
        else:
            self.points_per_link_object_collision_checking = int(num_interpolated_points_for_object_collision_checking / len(link_names_for_object_collision_checking))
        self.self_collision_margin_robot = self_collision_margin_robot
        self.num_interpolated_points_for_object_collision_checking = num_interpolated_points_for_object_collision_checking
        self.link_names_for_object_collision_checking = link_names_for_object_collision_checking
        self.n_links_for_object_collision_checking = len(link_names_for_object_collision_checking)
        self.link_margins_for_object_collision_checking = link_margins_for_object_collision_checking
        self.link_margins_for_object_collision_checking_robot_tensor = to_torch(
            link_margins_for_object_collision_checking, **self.tensor_args).repeat_interleave(
            int(num_interpolated_points_for_object_collision_checking / len(link_margins_for_object_collision_checking))
        )

        self.link_margins_for_object_collision_checking_tensor = self.link_margins_for_object_collision_checking_robot_tensor
        # append grasped object margins
        if self.grasped_object is not None:
            self.link_margins_for_object_collision_checking_tensor = torch.cat(
                (self.link_margins_for_object_collision_checking_tensor,
                 torch.ones(self.grasped_object.n_base_points_for_collision, **self.tensor_args) * self.margin_for_grasped_object_collision_checking)
            )

        self.link_idxs_for_object_collision_checking = link_idxs_for_object_collision_checking

        ################################################################################################
        # Self collision field
        if link_names_for_self_collision_checking is None:
            self.df_collision_self = None
        else:
            assert num_interpolated_points_for_self_collision_checking >= len(link_names_for_self_collision_checking)
            if num_interpolated_points_for_self_collision_checking % len(link_names_for_self_collision_checking) != 0:
                self.points_per_link_self_collision_checking = ceil(num_interpolated_points_for_self_collision_checking / len(link_names_for_self_collision_checking))
                num_interpolated_points_for_self_collision_checking = self.points_per_link_self_collision_checking * len(link_names_for_self_collision_checking)
            else:
                self.points_per_link_self_collision_checking = int(num_interpolated_points_for_self_collision_checking / len(link_names_for_self_collision_checking))

            self.link_names_for_self_collision_checking = link_names_for_self_collision_checking
            self.link_names_pairs_for_self_collision_checking = link_names_pairs_for_self_collision_checking

            self.link_idxs_for_self_collision_checking = link_idxs_for_self_collision_checking

            self.link_names_for_self_collision_checking_with_grasped_object = link_names_for_self_collision_checking_with_grasped_object

            self.self_collision_margin_grasped_object = self_collision_margin_grasped_object

            # build indices to retrieve distances from self collision distance matrix
            # including the grasped object
            idxs_links_distance_matrix = []
            p = self.points_per_link_self_collision_checking
            total_self_distances_robot = 0
            for i, link_1 in enumerate(self.link_names_for_self_collision_checking):
                if link_1 in self.link_names_pairs_for_self_collision_checking:
                    for link_2 in self.link_names_pairs_for_self_collision_checking[link_1]:
                        j = self.link_names_for_self_collision_checking.index(link_2)
                        idxs = [(i*p + m, j*p + n) for m, n in list(itertools.product(range(p), range(p)))]
                        idxs_links_distance_matrix.extend(idxs)
                        total_self_distances_robot += len(idxs)

            self_collision_margin_vector = [self.self_collision_margin_robot] * total_self_distances_robot

            if self.grasped_object is not None:
                total_self_distances_grasped_object = 0
                self_collision_robot_last_row_idx = len(self.link_names_for_self_collision_checking) * p
                n_grasped_points = self.grasped_object.n_base_points_for_collision
                for link_1 in self.link_names_for_self_collision_checking_with_grasped_object:
                    j = self.link_names_for_self_collision_checking.index(link_1)
                    idxs = [(self_collision_robot_last_row_idx + m, j * p + n) for m, n in list(itertools.product(range(n_grasped_points), range(p)))]
                    idxs_links_distance_matrix.extend(idxs)
                    total_self_distances_grasped_object += len(idxs)

                self_collision_margin_vector.extend([self.self_collision_margin_grasped_object] * total_self_distances_grasped_object)

            self_collision_margin_vector = to_torch(self_collision_margin_vector, **self.tensor_args)

            self.df_collision_self = CollisionSelfField(
                self,
                link_idxs_for_collision_checking=self.link_idxs_for_self_collision_checking,
                idxs_links_distance_matrix=idxs_links_distance_matrix,
                num_interpolated_points=num_interpolated_points_for_self_collision_checking,
                cutoff_margin=self_collision_margin_vector,
                tensor_args=self.tensor_args
            )

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def get_position(self, x):
        return x[..., :self.q_dim]

    def get_velocity(self, x):
        vel = x[..., self.q_dim:2 * self.q_dim]
        # If there is no velocity in the state, then compute it via finite difference
        if x.nelement() != 0 and vel.nelement() == 0:
            vel = finite_difference_vector(x, dt=self.dt, method='central')
            return vel
        return vel

    def get_acceleration(self, x):
        acc = x[..., 2 * self.q_dim:3 * self.q_dim]
        # If there is no acceleration in the state, then compute it via finite difference
        if x.nelement() != 0 and acc.nelement() == 0:
            vel = self.get_velocity(x)
            acc = finite_difference_vector(vel, dt=self.dt, method='central')
            return acc
        return acc

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def fk_map_collision(self, q, **kwargs):
        q_original_shape = q.shape
        if len(q_original_shape) == 1:
            q = q.unsqueeze(0)  # add batch dimension
        task_space_positions = self.fk_map_collision_impl(q, **kwargs)
        return task_space_positions

    @abc.abstractmethod
    def fk_map_collision_impl(self, q, **kwargs):
        # q: (..., q_dim)
        # return: (..., links_collision_positions, 3)
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
