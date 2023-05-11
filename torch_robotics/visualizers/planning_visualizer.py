from matplotlib import pyplot as plt


class PlanningVisualizer:

    def __init__(self, env=None, robot=None, planner=None):
        self.env = env
        self.robot = robot
        self.planner = planner

    def render_trajectory(self, traj, start_state=None, goal_state=None):
        fig = plt.figure(layout='tight')
        if self.env.dim == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()

        self.planner.render(ax)
        self.env.render(ax)
        self.robot.render_trajectory(ax, traj, start_state, goal_state)
        ax.set_aspect('equal')
        return fig, ax
