import einops
import numpy as np
import torch

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_rot_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


class RobotPanda(RobotBase):

    def __init__(self,
                 gripper=False,
                 tensor_args=None,
                 **kwargs):

        self.gripper = gripper

        #############################################
        # Differentiable robot model
        self.diff_panda = DifferentiableFrankaPanda(gripper=gripper, device=tensor_args['device'])
        self.link_names_for_collision_checking = [
            'panda_link1', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link7',
            'panda_hand', 'ee_link'
        ]
        self.link_name_ee = 'ee_link'

        self.jl_lower, self.jl_upper, _, _ = self.diff_panda.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        super().__init__(
            name='RobotPanda',
            q_limits=q_limits,
            num_interpolate=4,
            link_interpolate_range=[2, 7],
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

        if len(q_orig_shape) == 3:
            link_tensor = einops.rearrange(link_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        if pos_only:
            link_pos = link_pos_from_link_tensor(link_tensor)  # (batch horizon), taskspaces, x_dim
            return link_pos
        else:
            return link_tensor

    def render(self, ax, q=None, color='blue', arrow_length=0.15, arrow_alpha=1.0, arrow_linewidth=2.0, **kwargs):
        # draw skeleton
        skeleton = get_skeleton_from_model(self.diff_panda, q, self.diff_panda.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # draw EE frame
        H_EE = self.diff_panda.compute_forward_kinematics_all_links(
            q.unsqueeze(0), link_list=[self.link_name_ee]
        )
        frame_EE = Frame(
            rot=link_rot_from_link_tensor(H_EE).squeeze(),
            trans=link_pos_from_link_tensor(H_EE).squeeze(),
            device=self.tensor_args['device']
        )
        plot_coordinate_frame(
            ax, frame_EE, tensor_args=self.tensor_args,
            arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth
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
