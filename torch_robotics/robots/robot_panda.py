from collections import OrderedDict
from functools import partial
from itertools import product

import einops
import numpy as np
import torch
import yaml
from torch import vmap

from torch_robotics.environments.primitives import MultiSphereField
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_convert_wxyz
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_rot_from_link_tensor, \
    link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path
from torch_robotics.torch_planning_objectives.fields.distance_fields import interpolate_points_v1, CollisionSelfFieldWrapperSTORM
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


def transform_spheres_centers(x, link_tensor):
    H_blockdiag = torch.block_diag(*link_tensor)
    x_trans = H_blockdiag @ x
    return x_trans


class RobotPanda(RobotBase):

    def __init__(self,
                 use_self_collision_storm=False,
                 use_collision_spheres=True,
                 grasped_object=None,
                 tensor_args=None,
                 **kwargs):

        self.gripper = False

        ##########################################################################################
        # Differentiable robots model
        self.link_name_ee = 'ee_link'
        self.link_name_grasped_object = 'grasped_object'

        self.diff_panda = DifferentiableFrankaPanda(
            gripper=self.gripper, device=tensor_args['device'], grasped_object=grasped_object,
            use_collision_spheres=use_collision_spheres
        )

        self.jl_lower, self.jl_upper, _, _ = self.diff_panda.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        ##########################################################################################
        # Robot collision model for object avoidance
        self.use_collision_spheres = use_collision_spheres

        # OLD IMPLEMENTATION with block diagonal matrices
        # if use_collision_spheres:
        #     # Use a sphere collision model
        #     robot_collision_params = {
        #         'link_objs': [
        #             'panda_link2',
        #             'panda_link3',
        #             'panda_link4',
        #             'panda_link5',
        #             'panda_link6',
        #             # 'panda_link7',
        #             'panda_hand'
        #         ],
        #         'collision_spheres': 'panda/panda_sphere_config.yaml'
        #     }
        #     self.robot_links = robot_collision_params['link_objs']
        #
        #     # load collision file:
        #     coll_yml = (get_configs_path() / robot_collision_params['collision_spheres']).as_posix()
        #     with open(coll_yml) as file:
        #         coll_params = yaml.load(file, Loader=yaml.FullLoader)
        #
        #     spheres_centers_base = []
        #     spheres_centers_radii = []
        #     self.select_from_block_diag_idxs = []
        #     n_spheres_total = 0
        #     link_idxs_for_object_collision_checking = []
        #     for j_idx, j in enumerate(self.robot_links):
        #         idx = self.diff_panda._name_to_idx_map[j]
        #         link_idxs_for_object_collision_checking.append(idx)
        #         n_spheres = len(coll_params[j])
        #         for n in np.arange(n_spheres_total, n_spheres_total + n_spheres):
        #             for m in np.arange(j_idx * 4, (j_idx+1) * 4):
        #                 self.select_from_block_diag_idxs.append([m, n])
        #         n_spheres_total += n_spheres
        #         link_spheres = torch.ones((n_spheres, 4)).to(**tensor_args)  # homogeneous coordinate
        #         for i in range(n_spheres):
        #             link_spheres[i, :3] = torch.tensor(coll_params[j][i][:3], **tensor_args)
        #             spheres_centers_radii.append(coll_params[j][i][3])
        #
        #         spheres_centers_base.append(link_spheres)
        #
        #     self.spheres_centers_base_block_diag = torch.block_diag(*spheres_centers_base)
        #     self.spheres_centers_base_block_diag_transposed = torch.transpose(self.spheres_centers_base_block_diag, -2, -1)
        #     self.spheres_centers_radii = torch.tensor(spheres_centers_radii, **tensor_args)
        #
        #     self.select_from_block_diag_idxs = np.array(self.select_from_block_diag_idxs)
        #
        #     link_names_for_object_collision_checking = self.robot_links
        #     link_margins_for_object_collision_checking = self.spheres_centers_radii

        if use_collision_spheres:
            link_names_for_object_collision_checking = self.diff_panda.link_collision_names
            link_margins_for_object_collision_checking = to_torch(self.diff_panda.link_collision_margins, **tensor_args).view(-1, 1)

            assert len(link_names_for_object_collision_checking) == len(link_margins_for_object_collision_checking)

        else:
            # Use a simpler object collision model
            # Linearly interpolates between links, placing spheres with different radius
            # https://media.cheggcdn.com/media%2Fce1%2Fce100d57-2fdf-4cd7-8f4a-111a156d6339%2Fphp1EL2S4.png
            # https://www.researchgate.net/profile/Jesse-Haviland/publication/361785335/figure/fig1/AS:1174695604953098@1657080665902/The-Elementary-Transform-Sequence-of-the-7-degree-offreedom-Franka-Emika-Panda.png
            link_names_for_object_collision_checking = [
                # 'panda_link0',
                # 'panda_link1',
                'panda_link2',
                'panda_link3',
                # 'panda_link4',
                'panda_link5',
                # 'panda_link6',
                'panda_link7',
                'panda_hand',
                # self.link_name_ee,
            ]
            # these margins correspond to link_names_for_collision_checking
            link_margins_for_object_collision_checking = [
                # 0.1,
                # 0.1,
                0.125,
                0.125,
                # 0.075,
                0.13,
                # 0.1,
                0.1,
                0.08,
                # 0.025,
            ]
            assert len(link_names_for_object_collision_checking) == len(link_margins_for_object_collision_checking)

        link_idxs_for_object_collision_checking = []
        for link_name in link_names_for_object_collision_checking:
            idx = self.diff_panda._name_to_idx_map[link_name]
            link_idxs_for_object_collision_checking.append(idx)

        ##########################################################################################
        # Robot collision model for self collision
        link_names_pairs_for_self_collision_checking = OrderedDict({
            'panda_hand': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link6': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link5': ['panda_link0', 'panda_link1', 'panda_link2'],
            'panda_link4': ['panda_link1']
        })

        # self collision due to grasped object
        link_names_for_self_collision_checking_with_grasped_object = [
            'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
            # 'panda_link5'
        ]

        # retrieve unique names
        # link_names_for_self_collision_checking = self.diff_panda.get_link_names()
        link_names_for_self_collision_checking = []
        for k, v in link_names_pairs_for_self_collision_checking.items():
            link_names_for_self_collision_checking.append(k)
            link_names_for_self_collision_checking.extend(v)
        link_names_for_self_collision_checking.extend(link_names_for_self_collision_checking_with_grasped_object)
        link_names_for_self_collision_checking = sorted(list(set(link_names_for_self_collision_checking)))

        link_idxs_for_self_collision_checking = []
        for link_name in link_names_for_self_collision_checking:
            idx = self.diff_panda._name_to_idx_map[link_name]
            link_idxs_for_self_collision_checking.append(idx)

        ##########################################################################################
        super().__init__(
            name='RobotPanda',
            q_limits=q_limits,
            grasped_object=grasped_object,
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            margin_for_grasped_object_collision_checking=0.001,  # small margin for object placement
            num_interpolated_points_for_object_collision_checking=len(link_names_for_object_collision_checking) * 1,
            link_names_for_self_collision_checking=link_names_for_self_collision_checking,
            link_names_pairs_for_self_collision_checking=link_names_pairs_for_self_collision_checking,
            link_idxs_for_self_collision_checking=link_idxs_for_self_collision_checking,
            num_interpolated_points_for_self_collision_checking=25,
            self_collision_margin_robot=0.05,
            link_names_for_self_collision_checking_with_grasped_object=link_names_for_self_collision_checking_with_grasped_object,
            self_collision_margin_grasped_object=0.05,
            use_collision_spheres=use_collision_spheres,
            tensor_args=tensor_args,
            **kwargs
        )

        #############################################
        # Override self collision distance field with the one from STORM - https://arxiv.org/abs/2104.13542
        if use_self_collision_storm:
            assert grasped_object is None, ("STORM self collision model does not work if objects are grasped. "
                                            "Learn a self collision model of the robots grasping the object "
                                            "(e.g. using the object mesh).")
            self.df_collision_self = CollisionSelfFieldWrapperSTORM(
                self, 'robot_self/franka_self_sdf.pt', self.q_dim, tensor_args=self.tensor_args)

    def fk_map_collision_impl(self, q, **kwargs):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        link_pose_dict = self.diff_panda.compute_forward_kinematics_all_links(q, return_dict=True)
        # link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.link_names_for_object_collision_checking)
        link_all_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_panda.get_link_names())

        # Transform collision points of the grasp object with the forward kinematics
        grasped_object_points_in_robot_base_frame = None
        if self.grasped_object:
            grasped_object_points_in_object_frame = self.grasped_object.base_points_for_collision
            frame_grasped_object = link_pose_dict[self.link_name_grasped_object]
            # TODO - by default assumes that world frame is the robots base frame
            grasped_object_points_in_robot_base_frame = frame_grasped_object.transform_point(grasped_object_points_in_object_frame)

        if len(q_orig_shape) == 3:
            link_all_tensor = einops.rearrange(link_all_tensor, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        # if self.use_collision_spheres:
        #     get the collision spheres centers
            # link_pos = self.get_spheres_centers(link_all_tensor)
        # else:

        link_pos = link_pos_from_link_tensor(link_all_tensor)  # (batch horizon), taskspaces, x_dim
        if grasped_object_points_in_robot_base_frame is not None:
            if len(q_orig_shape) == 3:
                grasped_object_points_in_robot_base_frame = einops.rearrange(grasped_object_points_in_robot_base_frame, "(b h) d1 d2 -> b h d1 d2", b=b, h=h)
            link_pos = torch.cat((link_pos, grasped_object_points_in_robot_base_frame), dim=-2)

        return link_pos

    # def get_spheres_centers(self, link_all_tensor):
    #     get only the needed link transformations
        # link_tensor = link_all_tensor[..., self.link_idxs_for_object_collision_checking, :, :]
        # new_centers = vmap(vmap(partial(transform_spheres_centers, self.spheres_centers_base_block_diag_transposed)))(link_tensor)
        # new_centers = new_centers[..., self.select_from_block_diag_idxs[:, 0], self.select_from_block_diag_idxs[:, 1]]
        # new_centers = torch.transpose(new_centers.reshape(*new_centers.shape[0:-1], -1, 4), -2, -1)
        # new_centers = torch.transpose(new_centers, -2, -1)[..., :3]  # take the position from homegenous coordinate
        # return new_centers

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

    def render(self, ax, q=None, color='blue', arrow_length=0.15, arrow_alpha=1.0, arrow_linewidth=2.0,
               draw_links_spheres=False, **kwargs):
        # draw skeleton
        skeleton = get_skeleton_from_model(self.diff_panda, q, self.diff_panda.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # forward kinematics
        fks_dict = self.diff_panda.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)

        # draw link collision points
        if draw_links_spheres:
            link_tensor = convert_link_dict_to_tensor(fks_dict, self.link_names_for_object_collision_checking)
            link_pos = link_pos_from_link_tensor(link_tensor)
            link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points_for_object_collision_checking).squeeze(0)
            spheres = MultiSphereField(
                link_pos,
                self.link_margins_for_object_collision_checking_robot_tensor.view(-1, 1),
                tensor_args=self.tensor_args)
            spheres.render(ax, color='red', cmap='Reds', **kwargs)

        # draw EE frame
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

            # draw object collision points
            points_in_object_frame = self.grasped_object.base_points_for_collision
            points_in_robot_base_frame = frame_grasped_object.transform_point(points_in_object_frame).squeeze()
            points_in_robot_base_frame_np = to_numpy(points_in_robot_base_frame)
            ax.scatter(
                points_in_robot_base_frame_np[:, 0],
                points_in_robot_base_frame_np[:, 1],
                points_in_robot_base_frame_np[:, 2],
                color=color
            )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.)
            if start_state is not None:
                self.render(ax, start_state, color='green')
            if goal_state is not None:
                self.render(ax, goal_state, color='purple')
