# MIT License

# Copyright (c) 2022 An Thai Le

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# **********************************************************************
# The first version of some files were licensed as
# "Original Source License" (see below). Several enhancements and bug fixes
# were done by An Thai Le since obtaining the first version.



# Original Source License:

# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
import torch
import numpy as np
import pinocchio as pin

from torch_kinematics_tree.models.rigid_body import DifferentiableRigidBody
from torch_kinematics_tree.models.utils import URDFRobotModel, MJCFRobotModel, convert_link_dict_to_tensor
from torch_kinematics_tree.models.pinocchio.model import PinocchioModel
from torch_kinematics_tree.models.pinocchio.tasks import BodyTask
from torch_kinematics_tree.geometrics.spatial_vector import MotionVec



class DifferentiableTree(torch.nn.Module):

    def __init__(self, model_path, name="", link_list=None, device='cpu'):

        super().__init__()

        self.name = name
        self.link_list = link_list

        self._device = device
        self.model_type = model_path.split('.')[-1]
        if self.model_type == 'urdf':
            self._model = URDFRobotModel(model_path=model_path, device=self._device)
        elif self.model_type == 'xml':
            self._model = MJCFRobotModel(model_path=model_path, device=self._device)
        else:
            raise NotImplementedError(f'{self.model_type} is not supported!')
        self._bodies = torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []

        self.pin_model = PinocchioModel(model_path=model_path)

        # NOTE: making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # NOTE: joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self._model.links):
            # Initialize body object
            rigid_body_params = self._model.get_body_parameters(i, link)
            body = DifferentiableRigidBody(
                rigid_body_params=rigid_body_params, device=self._device
            )
            if i == 0:
                body.is_root = True  # set root

            # Joint properties
            body.joint_idx = None
            if rigid_body_params["joint_type"] != "fixed":
                body.joint_idx = self._n_dofs
                self._n_dofs += 1
                self._controlled_joints.append(i)

            # Add to data structures
            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i

        # Once all bodies are loaded, connect each body to its parent
        for body in self._bodies[1:]:
            parent_body_name = self._model.get_name_of_parent_body(body.name)
            parent_body_idx = self._name_to_idx_map[parent_body_name]
            body.set_parent(self._bodies[parent_body_idx])
            self._bodies[parent_body_idx].add_child(body)

    def update_base_pose(self, pose_vec):
        self._bodies[0].update_pose(pose_vec)

    def update_kinematic_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        """

        Updates the kinematic state of the robot in a stateful way
        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]

        Returns:

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs

        batch_size = q.shape[0]

        # update the state of the joints
        for i in range(q.shape[1]):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_state(
                q[:, i].unsqueeze(1), qd[:, i].unsqueeze(1)
            )

        # we assume a non-moving base
        parent_body = self._bodies[0]
        parent_body.vel = MotionVec(
            torch.zeros((batch_size, 3), device=self._device),
            torch.zeros((batch_size, 3), device=self._device),
        )

        # propagate the new joint state through the kinematic chain to update bodies position/velocities
        for i in range(1, len(self._bodies)):

            body = self._bodies[i]
            parent_name = self._model.get_name_of_parent_body(body.name)
            # find the joint that has this link as child
            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # transformation operator from child link to parent link
            childToParentT = body.joint_pose
            # transformation operator from parent link to child link
            parentToChildT = childToParentT.inverse()

            # the position and orientation of the body in world coordinates, with origin at the joint
            body.pose = parent_body.pose.multiply_transform(childToParentT)

            # we rotate the velocity of the parent's body into the child frame
            new_vel = parent_body.vel.transform(parentToChildT)

            # this body's angular velocity is combination of the velocity experienced at it's parent's link
            # + the velocity created by this body's joint
            body.vel = body.joint_vel.add_motion_vec(new_vel)

        return

    def compute_forward_kinematics_all_links(
        self, q: torch.Tensor, return_dict=False,
    ) -> torch.Tensor:
        """
        Stateless forward kinematics
        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: dict link frame

        """
        batch_size = q.shape[0]
        # Create joint state dictionary
        q_dict = {}
        for i, body_idx in enumerate(self._controlled_joints):
            q_dict[self._bodies[body_idx].name] = q[:, i].unsqueeze(1)

        # Call forward kinematics on root node
        pose_dict = self._bodies[0].forward_kinematics(q_dict, batch_size)
        if not return_dict:
            return convert_link_dict_to_tensor(pose_dict, self.get_link_names())
        else:
            return pose_dict

    def compute_forward_kinematics_link_list(
        self, q: torch.Tensor, return_dict=False, link_list=None,
    ) -> torch.Tensor:
        """

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: dict link frame

        """
        batch_size = q.shape[0]
        if link_list is None:
            link_list = self.link_list

        # Create joint state dictionary
        q_dict = {}
        for i, body_idx in enumerate(self._controlled_joints):
            q_dict[self._bodies[body_idx].name] = q[:, i].unsqueeze(1)

        # Call forward kinematics on root node
        pose_dict = self._bodies[0].forward_kinematics(q_dict, batch_size)

        if link_list is None:  # link list is None again, output whole body
            link_list = self.get_link_names()
        if not return_dict:
            return convert_link_dict_to_tensor(pose_dict, link_list)
        else:
            link_dict = {k: v for k, v in pose_dict.items() if k in link_list}
            return link_dict

    def compute_forward_kinematics(
        self, q: torch.Tensor,  qd: torch.Tensor, link_name: str, state_less: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        assert q.ndim == 2

        if state_less:
            return self.compute_forward_kinematics_all_links(q)[link_name]  # return SE3 matrix
        else:
            self.update_kinematic_state(q, qd)

            pose = self._bodies[self._name_to_idx_map[link_name]].pose
            pos = pose.translation
            rot = pose.get_quaternion()
            return pos, rot  # return tuple of translation and rotation

    def compute_fk_and_jacobian(
        self, q: torch.Tensor, qd: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
        Returns: ee_pos, ee_rot and linear and angular jacobian
        """
        batch_size = q.shape[0]
        ee_pos, ee_rot = self.compute_forward_kinematics(q, qd, link_name)

        lin_jac = torch.zeros([batch_size, 3, self._n_dofs], device=self._device)
        ang_jac = torch.zeros([batch_size, 3, self._n_dofs], device=self._device)
        # any joints larger than this joint, will have 0 in the jacobian
        # parent_name = self._urdf_model.get_name_of_parent_body(link_name)
        parent_joint_id = self._model.find_joint_of_body(link_name)

        # do this as a tensor cross product:
        for i, idx in enumerate(self._controlled_joints):
            if (idx - 1) > parent_joint_id:
                continue
            pose = self._bodies[idx].pose
            axis_idx = self._bodies[idx].axis_idx
            p_i = pose.translation
            z_i = torch.index_select(pose.rotation, -1, axis_idx).squeeze(-1)
            lin_jac[:, :, i] = torch.cross(z_i, ee_pos - p_i)
            ang_jac[:, :, i] = z_i

        return (
            ee_pos, ee_rot, lin_jac, ang_jac,
        )

    def inverse_kinematics(self, target_position, target_euler, frame='ee_link', step_rate=0.05, **kwargs):
        """
        Inverse kinematics
        Args:
            target_position (np.array): target position
            target_euler (np.array): target orientation
            joint_indices (np.array): joint indices

        Returns:
            np.array: joint angles
        """
        target_orientation = Rotation.from_euler('xyz', target_euler).as_matrix()
        pose = pin.SE3(target_orientation, target_position)
        end_effector_task = BodyTask(
            frame,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1.0,  # tuned for this setup
        )
        end_effector_task.set_target(pose)
        return self.pin_model.solve_ik([end_effector_task], dt=step_rate, **kwargs)

    def get_joint_limits(self) -> List[Dict[str, torch.Tensor]]:
        """

        Returns: list of joint limit dict, containing joint position, velocity and effort limits

        """
        limits = []
        for idx in self._controlled_joints:
            limits.append(self._bodies[idx].get_joint_limits())
        return limits

    def get_joint_limit_array(self) -> Tuple[np.ndarray]:
        """

        Returns: list of joint limit dict, containing joint position, velocity and effort limits

        """
        lowers = []
        uppers = []
        vel_lowers = []
        vel_uppers = []
        for idx in self._controlled_joints:
            data = self._bodies[idx].get_joint_limits()
            lowers.append(data['lower'])
            uppers.append(data['upper'])
            vel_lowers.append(-data['velocity'])
            vel_uppers.append(data['velocity'])
        return np.array(lowers), np.array(uppers), np.array(vel_lowers), np.array(vel_uppers)

    def get_link_names(self) -> List[str]:
        """

        Returns: a list containing names for all links

        """

        link_names = []
        for i in range(len(self._bodies)):
            link_names.append(self._bodies[i].name)
        return link_names

    def print_link_names(self) -> None:
        """

        print the names of all links

        """
        names = self.get_link_names()
        for i in range(len(names)):
            print(names[i])
