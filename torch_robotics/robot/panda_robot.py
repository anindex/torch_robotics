import einops

from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_utils.torch_utils import to_numpy


class PandaRobot(RobotBase):

    def __init__(self,
                 **kwargs):
        super().__init__(
            num_interpolate=4,
            link_interpolate_range=[2, 7],
            **kwargs
        )

        #############################################
        # Differentiable robot model
        self.diff_panda = DifferentiableFrankaPanda(gripper=True, device=self.tensor_args['device'])
        self.link_names_for_collision_checking = [
            'panda_link1', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link7', 'panda_link8', 'ee_link'
        ]
        self.link_name_ee = 'ee_link'

    def fk_map_impl(self, q):
        b, h, d = q.shape
        q = einops.rearrange(q, 'b h d -> (b h) d')

        link_tensor = self.diff_panda.compute_forward_kinematics_link_list(
            q, link_list=self.link_names_for_collision_checking
        )

        # reshape to batch, trajectory, link poses
        link_tensor = einops.rearrange(link_tensor, '(b h) links d1 d2 -> b h links d1 d2', b=b, h=h)
        link_pos = link_pos_from_link_tensor(link_tensor)  # batch, horizon, taskspaces, x_dim
        x_pos = einops.rearrange(link_pos, '(b h) t d -> b h t d', b=b, h=h)
        return x_pos

    def render_trajectory(self, ax, traj=None, start_state=None, goal_state=None, **kwargs):
        raise NotImplementedError
