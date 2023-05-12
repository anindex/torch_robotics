from torch_robotics.robot.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy


class PointMassRobot(RobotBase):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def fk_map_impl(self, q, pos_only=False):
        # There is no forward kinematics. Assume it's the identity.
        # Add task space dimension
        if pos_only:
            return q.unsqueeze(-2)
        else:
            raise NotImplementedError

    def render(self, ax, q=None, color='blue', **kwargs):
        if q is not None:
            q = to_numpy(q)
            ax.scatter(*q, color=color)

    def render_trajectory(self, ax, traj_q=None, start_state=None, goal_state=None, **kwargs):
        if traj_q is not None:
            traj_np = to_numpy(traj_q)
            if self.q_n_dofs == 3:
                ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], 'b-', markersize=3)
            else:
                ax.plot(traj_np[:, 0], traj_np[:, 1], 'b-', markersize=3)
        if start_state is not None:
            start_state_np = to_numpy(start_state)
            if len(start_state_np) == 3:
                ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=7)
            else:
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=7)
        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            if len(goal_state_np) == 3:
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], 'ro', markersize=7)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], 'ro', markersize=7)
