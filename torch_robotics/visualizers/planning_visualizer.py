import os

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def create_fig_and_axes(dim=2):
    fig = plt.figure(layout='tight')
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    return fig, ax


class PlanningVisualizer:

    def __init__(self, env=None, robot=None, planner=None):
        self.env = env
        self.robot = robot
        self.planner = planner

    def render_trajectory(
            self, traj, render_planner=False,
            animate=False, video_filepath='movie_trajectory.mp4',
            **kwargs
    ):
        fig, ax = create_fig_and_axes(dim=self.env.dim)

        if render_planner:
            self.planner.render(ax)
        self.env.render(ax)
        self.robot.render_trajectory(ax, traj, **kwargs)

        if animate and traj is not None:
            fig2, ax2 = create_fig_and_axes(dim=self.env.dim)
            def animate_fn(i):
                ax2.clear()
                self.env.render(ax2)
                q = traj[i].squeeze()
                self.robot.render(ax2, q)

            anim_time_in_sec = 5
            str_create = "Creating animation"
            print(f'{str_create}...')
            ani = FuncAnimation(fig2, animate_fn,
                                frames=len(traj),
                                interval=anim_time_in_sec * 1000 / len(traj),
                                repeat=False)
            print(f'...finished {str_create}')

            print('Saving video...')
            ani.save(os.path.join(video_filepath), fps=int(len(traj) / anim_time_in_sec), dpi=90)
            print('...finished Saving video')

        return fig, ax
