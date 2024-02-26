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

from typing import List, Dict, Optional

import torch
from torch_robotics.torch_kinematics_tree.geometrics.spatial_vector import (
    MotionVec,
    z_rot,
    y_rot,
    x_rot,
)
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.utils import vector3_to_skew_symm_matrix, cross_product



class DifferentiableSpatialRigidBodyInertia(torch.nn.Module):
    def __init__(self, rigid_body_params, device="cpu"):
        super().__init__()
        self.mass = rigid_body_params["mass"]
        self.com = rigid_body_params["com"]
        self.inertia_mat = rigid_body_params["inertia_mat"]

        self.device = torch.device(device)

    def _get_parameter_values(self):
        return self.mass, self.com, self.inertia_mat

    def multiply_motion_vec(self, smv):
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = com * mass
        com_skew_symm_mat = vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )

        batch_size = smv.lin.shape[0]
        new_lin_force = mass * smv.lin - cross_product(
            mcom.repeat(batch_size, 1), smv.ang
        )
        new_ang_force = (
            inertia.repeat(batch_size, 1, 1) @ smv.ang.unsqueeze(2)
        ).squeeze(2) + cross_product(mcom.repeat(batch_size, 1), smv.lin)

        return MotionVec(new_lin_force, new_ang_force, device=self.device)

    def get_spatial_mat(self):
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = mass * com
        com_skew_symm_mat = vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )
        mat = torch.zeros((6, 6), device=self.device)
        mat[:3, :3] = inertia
        mat[3, 0] = 0
        mat[3, 1] = mcom[0, 2]
        mat[3, 2] = -mcom[0, 1]
        mat[4, 0] = -mcom[0, 2]
        mat[4, 1] = 0.0
        mat[4, 2] = mcom[0, 0]
        mat[5, 0] = mcom[0, 1]
        mat[5, 1] = -mcom[0, 0]
        mat[5, 2] = 0.0

        mat[0, 3] = 0
        mat[0, 4] = -mcom[0, 2]
        mat[0, 5] = mcom[0, 1]
        mat[1, 3] = mcom[0, 2]
        mat[1, 4] = 0.0
        mat[1, 5] = -mcom[0, 0]
        mat[2, 3] = -mcom[0, 1]
        mat[2, 4] = mcom[0, 0]
        mat[2, 5] = 0.0

        mat[3, 3] = mass
        mat[4, 4] = mass
        mat[5, 5] = mass
        return mat




class DifferentiableRigidBody(torch.nn.Module):
    """
    Differentiable Representation of a link
    """

    _parents: Optional["DifferentiableRigidBody"]
    _children: List["DifferentiableRigidBody"]

    def __init__(self, rigid_body_params, device="cpu"):

        super().__init__()

        self.is_root = False
        self._parents = None
        self._children = []

        self.device = device
        self.joint_id = rigid_body_params["joint_id"]
        self.name = rigid_body_params["link_name"]

        # dynamics parameters
        self.inertia = DifferentiableSpatialRigidBodyInertia(
            rigid_body_params, device=self.device
        )
        self.joint_damping = rigid_body_params["joint_damping"]

        self.trans = rigid_body_params["trans"].reshape(1, 3)
        self.rot_angles = rigid_body_params["rot_angles"].reshape(1, 3)
        rot_angles_vals = self.rot_angles
        roll = rot_angles_vals[0, 0]
        pitch = rot_angles_vals[0, 1]
        yaw = rot_angles_vals[0, 2]
        self.fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]
        self.axis_idx = torch.nonzero(self.joint_axis.squeeze(0))
        if self.axis_idx.nelement() > 0:
            self.axis_idx = self.axis_idx[0]

        if self.joint_axis[0, 0] == 1:
            self.axis_rot_fn = x_rot
        elif self.joint_axis[0, 1] == 1:
            self.axis_rot_fn = y_rot
        else:
            self.axis_rot_fn = z_rot
        self.joint_type = rigid_body_params["joint_type"]
        self.joint_limits = rigid_body_params["joint_limits"]

        self.joint_pose = Frame(device=self.device)
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        # local velocities and accelerations (w.r.t. joint coordinate frame):
        self.joint_vel = MotionVec(device=self.device)
        self.joint_acc = MotionVec(device=self.device)

        self.update_joint_state(
            torch.zeros([1, 1], device=self.device),
            torch.zeros([1, 1], device=self.device),
        )
        self.update_joint_acc(torch.zeros([1, 1], device=self.device))

        self.reset()

    def reset(self):
        '''
        Reset pose, velocity and acceleration to init state.
        '''
        self.pose = Frame(device=self.device)
        self.vel = MotionVec(device=self.device)
        self.acc = MotionVec(device=self.device)
        self.force = MotionVec(device=self.device)

    def update_pose(self, pose_vec):
        pose = Frame(device=self.device)
        pose.set_pose(pose_vec)
        self.pose = pose

    # Kinematic tree construction
    def set_parent(self, link: "DifferentiableRigidBody"):
        self._parent = link

    def add_child(self, link: "DifferentiableRigidBody"):
        self._children.append(link)

    # Recursive algorithms
    def forward_kinematics(self, q_dict, batch_size):
        """Recursive forward kinematics
        Computes transformations from self to all descendants.

        Returns: Dict[link_name, transform_from_self_to_link]
        """
        # Compute joint pose
        if self.name in q_dict:
            q = q_dict[self.name]

            # bound q with limits
            if self.joint_limits is not None:
                q_clamped = torch.clamp(q, min=self.joint_limits['lower'], max=self.joint_limits['upper'])
            else:
                q_clamped = q
            
            if self.joint_type in ['revolute', 'continuous']:
                if torch.abs(self.joint_axis[0, 0]) == 1:
                    rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q_clamped)
                elif torch.abs(self.joint_axis[0, 1]) == 1:
                    rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q_clamped)
                else:
                    rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q_clamped)

                joint_pose = Frame(
                    rot=self.fixed_rotation.repeat(batch_size, 1, 1) @ rot,
                    trans=self.trans.repeat(batch_size, 1),
                    device=self.device,
                )
            elif self.joint_type == 'prismatic':
                trans = self.joint_axis * q_clamped

                joint_pose = Frame(
                    rot=self.fixed_rotation.repeat(batch_size, 1, 1),
                    trans=self.trans + trans,
                    device=self.device,
                )
            else:
                raise NotImplementedError()
        else:
            joint_pose = Frame(
                rot=self.fixed_rotation.repeat(batch_size, 1, 1),
                trans=self.trans.repeat(batch_size, 1),
                device=self.device,
            )
            if self.is_root:
                joint_pose = self.pose.multiply_transform(joint_pose)

        # Compute forward kinematics of children
        pose_dict = {self.name: self.pose}
        for child in self._children:
            pose_dict.update(child.forward_kinematics(q_dict, batch_size))

        # Apply joint pose
        result = {}
        for body_name in pose_dict:
            if self.is_root and body_name == self.name:
                base_pose = Frame(
                    rot=self.pose.rotation.repeat(batch_size, 1, 1),
                    trans=self.pose.translation.repeat(batch_size, 1),
                    device=self.device,
                )
                result[body_name] = base_pose
            else:
                result[body_name] = joint_pose.multiply_transform(pose_dict[body_name])
        return result

    # Get/set
    def update_joint_state(self, q, qd):
        batch_size = q.shape[0]

        # bound q with limits
        if self.joint_limits is not None:
            q_clamped = torch.clamp(q, min=self.joint_limits['lower'], max=self.joint_limits['upper'])
            qd_clamped = qd  # not clamping velocity for now
            # qd_clamped = torch.clamp(qd, min=-self.joint_limits['velocity'], max=self.joint_limits['velocity'])
        else:
            q_clamped = q
            qd_clamped = qd

        if self.joint_type in ['revolute', 'continuous']:
            rot = self.axis_rot_fn(q_clamped)
            # if torch.abs(self.joint_axis[0, 0]) == 1:
            #     rot = x_rot(torch.sign(self.joint_axis[0, 0]) * q_clamped)
            # elif torch.abs(self.joint_axis[0, 1]) == 1:
            #     rot = y_rot(torch.sign(self.joint_axis[0, 1]) * q_clamped)
            # else:
            #     rot = z_rot(torch.sign(self.joint_axis[0, 2]) * q_clamped)

            self.joint_pose.set_translation(
                self.trans.repeat(batch_size, 1)
            )
            self.joint_pose.set_rotation(self.fixed_rotation.repeat(batch_size, 1, 1) @ rot)
        elif self.joint_type == 'prismatic':
            trans = self.joint_axis * q_clamped
            self.joint_pose.set_translation(
                self.trans + trans
            )
            self.joint_pose.set_rotation(self.fixed_rotation.repeat(batch_size, 1, 1))
        elif self.joint_type == 'fixed':
            self.joint_pose.set_translation(
                self.trans.repeat(batch_size, 1)
            )
            self.joint_pose.set_rotation(self.fixed_rotation.repeat(batch_size, 1, 1))
        else:
            raise NotImplementedError(f'Joint Type: {self.joint_type}')

        joint_ang_vel = qd_clamped @ self.joint_axis
        self.joint_vel = MotionVec(
            torch.zeros_like(joint_ang_vel), joint_ang_vel
        )

    def update_joint_acc(self, qdd):
        # local z axis (w.r.t. joint coordinate frame):
        joint_ang_acc = qdd @ self.joint_axis
        self.joint_acc = MotionVec(
            torch.zeros_like(joint_ang_acc), joint_ang_acc
        )

    # def multiply_inertia_with_motion_vec(self, motion_vec: MotionVec) -> MotionVec:

    #     mass, com, inertia_mat = self._get_dynamics_parameters_values()

    #     mcom = com * mass
    #     com_skew_symm_mat = vector3_to_skew_symm_matrix(com)
    #     inertia = inertia_mat + mass * (
    #         com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
    #     )

    #     return MotionVec(
    #         mass * motion_vec.lin - cross_product(mcom, motion_vec.ang),
    #         (inertia @ motion_vec.ang.unsqueeze(2)).squeeze(2)
    #         + cross_product(mcom, motion_vec.lin),
    #     )

    def _get_dynamics_parameters_values(self):
        return self.inertia._get_parameter_values()

    def get_joint_limits(self):
        return self.joint_limits

    def get_joint_damping_const(self):
        return self.joint_damping
