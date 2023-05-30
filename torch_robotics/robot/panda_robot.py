import einops
import torch

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda


class PandaRobot(RobotBase):

    def __init__(self,
                 tensor_args=None,
                 **kwargs):

        self.jl_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.jl_upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        q_limits = torch.tensor([self.jl_lower, self.jl_upper], **tensor_args)

        super().__init__(
            q_limits=q_limits,
            num_interpolate=4,
            link_interpolate_range=[2, 7],
            tensor_args=tensor_args,
            **kwargs
        )

        #############################################
        # Differentiable robot model
        self.diff_panda = DifferentiableFrankaPanda(gripper=False, device=self.tensor_args['device'])
        self.link_names_for_collision_checking = [
            'panda_link1', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link7', 'panda_link8', 'ee_link'
        ]
        self.link_name_ee = 'ee_link'

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

        link_tensor = self.diff_panda.compute_forward_kinematics_link_list(
            q, link_list=self.link_names_for_collision_checking
        )

        if len(q_orig_shape) == 3:
            link_tensor = einops.rearrange(link_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        if pos_only:
            link_pos = link_pos_from_link_tensor(link_tensor)  # (batch horizon), taskspaces, x_dim
            return link_pos
        else:
            return link_tensor

    def render(self, ax, q=None, color='blue', **kwargs):
        skeleton = get_skeleton_from_model(self.diff_panda, q, self.diff_panda.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            for traj, color in zip(trajs, colors):
                for t in range(traj.shape[0] - 1):
                    skeleton = get_skeleton_from_model(self.diff_panda, traj[t], self.diff_panda.get_link_names())
                    skeleton.draw_skeleton(ax=ax, color=color)
            if start_state is not None:
                start_skeleton = get_skeleton_from_model(self.diff_panda, start_state, self.diff_panda.get_link_names())
                start_skeleton.draw_skeleton(ax=ax, color='blue')
            if goal_state is not None:
                start_skeleton = get_skeleton_from_model(self.diff_panda, goal_state, self.diff_panda.get_link_names())
                start_skeleton.draw_skeleton(ax=ax, color='red')
