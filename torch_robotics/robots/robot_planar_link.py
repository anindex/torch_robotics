import os

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS


class RobotPlanar2Link(RobotBase):

    link_name_ee = 'link_ee'  # must be in the urdf file

    def __init__(self,
                 urdf_robot_file=os.path.join(get_robot_path(), "planar_robot_2_link.urdf"),
                 collision_spheres_file_path=os.path.join(
                     get_configs_path(), 'planar_robot_2_link/planar_robot_2_link_sphere_config.yaml'),
                 task_space_dim=2,
                 **kwargs):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            link_name_ee=self.link_name_ee,
            task_space_dim=task_space_dim,
            **kwargs
        )

        ################################################################################################
        # Link indices for rendering
        self.link_idxs = self.get_link_idxs_for_rendering()

    def get_link_idxs_for_rendering(self):
        return [
            self.robot_torchkin.link_map['link_2'].id,
            self.robot_torchkin.link_map[self.link_name_ee].id
        ]

    def render(self, ax, q=None, alpha=1.0, color='blue', linewidth=2.0, **kwargs):
        H_all = self.fk_all(q.unsqueeze(0))
        p_all = [link_pos_from_link_tensor(H_all[idx]).squeeze() for idx in self.link_idxs]
        p_all = to_numpy([to_numpy(p) for p in p_all])
        ax.plot([0, p_all[0][0]], [0, p_all[0][1]], color=color, linewidth=linewidth, alpha=alpha)
        for p1, p2 in zip(p_all[:-1], p_all[1:]):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter(p_all[-1][0], p_all[-1][1], color='red', marker='o')

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for _trajs_pos in trajs_pos:
                for q, color in zip(_trajs_pos, colors):
                    self.render(ax, q, alpha=0.8, color=color)
        if start_state is not None:
            self.render(ax, start_state, alpha=1.0, color='blue')
        if goal_state is not None:
            self.render(ax, goal_state, alpha=1.0, color='red')


class RobotPlanar4Link(RobotPlanar2Link):

    def __init__(self,
                 urdf_robot_file=os.path.join(get_robot_path(), "planar_robot_4_link.urdf"),
                 collision_spheres_file_path=os.path.join(
                     get_configs_path(), 'planar_robot_4_link/planar_robot_4_link_sphere_config.yaml'),
                 **kwargs):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            **kwargs
        )

    def get_link_idxs_for_rendering(self):
        return [
            self.robot_torchkin.link_map['link_2'].id,
            self.robot_torchkin.link_map['link_3'].id,
            self.robot_torchkin.link_map['link_4'].id,
            self.robot_torchkin.link_map[self.link_name_ee].id
        ]


if __name__ == '__main__':
    robot = RobotPlanar2Link(tensor_args=DEFAULT_TENSOR_ARGS)
    robot = RobotPlanar4Link(tensor_args=DEFAULT_TENSOR_ARGS)
