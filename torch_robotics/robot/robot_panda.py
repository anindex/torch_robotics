import einops
import numpy as np
import torch

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_convert_wxyz
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_rot_from_link_tensor, \
    link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


class RobotPanda(RobotBase):

    def __init__(self,
                 grasped_object=None,
                 tensor_args=None,
                 **kwargs):

        self.gripper = False

        #############################################
        # Differentiable robot model
        self.grasped_object = grasped_object

        self.link_name_ee = 'ee_link'
        self.link_name_grasped_object = 'grasped_object'

        self.link_names_for_collision_checking = [
            'panda_link1', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link7',
            'panda_hand',
            self.link_name_grasped_object if grasped_object else self.link_name_ee,
        ]

        self.diff_panda = DifferentiableFrankaPanda(
            gripper=self.gripper, device=tensor_args['device'], grasped_object=grasped_object
        )

        self.jl_lower, self.jl_upper, _, _ = self.diff_panda.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        super().__init__(
            name='RobotPanda',
            q_limits=q_limits,
            num_interpolate=4,
            link_interpolate_range=[0, len(self.link_names_for_collision_checking)-1],  # which links to interpolate for collision checking
            tensor_args=tensor_args,
            **kwargs
        )

    def fk_map_impl(self, q, pos_only=False):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        link_tensor = self.diff_panda.compute_forward_kinematics_all_links(
            q, link_list=self.link_names_for_collision_checking
        )

        # Transform points of the grasp object with the forward kinematics

        if len(q_orig_shape) == 3:
            link_tensor = einops.rearrange(link_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        if pos_only:
            link_pos = link_pos_from_link_tensor(link_tensor)  # (batch horizon), taskspaces, x_dim
            return link_pos
        else:
            return link_tensor

    def get_EE_pose(self, q):
        return self.diff_panda.compute_forward_kinematics_all_links(q, link_list=[self.link_name_ee])

    def get_EE_position(self, q):
        ee_pose = self.get_EE_pose(q)
        return link_pos_from_link_tensor(ee_pose)

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        if rotation_matrix:
            return link_rot_from_link_tensor(ee_pose)
        else:
            return link_quat_from_link_tensor(ee_pose)

    def render(self, ax, q=None, color='blue', arrow_length=0.15, arrow_alpha=1.0, arrow_linewidth=2.0, **kwargs):
        # draw skeleton
        skeleton = get_skeleton_from_model(self.diff_panda, q, self.diff_panda.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # draw EE frame
        fks_dict = self.diff_panda.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)
        frame_EE = fks_dict[self.link_name_ee]
        plot_coordinate_frame(
            ax, frame_EE, tensor_args=self.tensor_args,
            arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth
        )

        # draw grasped object
        if self.grasped_object is not None:
            frame_grasped_object = fks_dict[self.link_name_grasped_object]

            # draw object
            pos = frame_grasped_object.translation.squeeze()
            ori = q_convert_wxyz(frame_grasped_object.get_quaternion().squeeze())
            self.grasped_object.render(ax, pos=pos, ori=ori, color=color)

            # draw points in object for collision
            points_in_object_frame = self.grasped_object.base_points_for_collision
            points_in_robot_base_frame = frame_grasped_object.transform_point(points_in_object_frame)
            points_in_robot_base_frame_np = to_numpy(points_in_robot_base_frame)
            ax.scatter(
                points_in_robot_base_frame_np[:, 0],
                points_in_robot_base_frame_np[:, 1],
                points_in_robot_base_frame_np[:, 2],
                color=color
            )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            for traj, color in zip(trajs, colors):
                for t in range(traj.shape[0] - 1):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.)
            if start_state is not None:
                self.render(ax, start_state, color='green')
            if goal_state is not None:
                self.render(ax, start_state, color='purple')
