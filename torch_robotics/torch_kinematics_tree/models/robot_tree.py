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
import torch
import numpy as np
from torch.autograd.functional import jacobian

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_axis_angles, q_div, q_convert_wxyz
from torch_robotics.torch_kinematics_tree.models.rigid_body import DifferentiableRigidBody
from torch_robotics.torch_kinematics_tree.models.utils import URDFRobotModel
from torch_robotics.torch_kinematics_tree.geometrics.spatial_vector import MotionVec
from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance, link_pos_from_link_tensor, \
    link_quat_from_link_tensor, link_rot_from_link_tensor, cross_product
from torch_robotics.torch_utils.torch_utils import to_torch, torch_intersect_1d


def convert_link_dict_to_tensor(link_dict, link_list):
    return torch.stack([link_dict[name].get_transform_matrix() for name in link_list], dim=1)


class DifferentiableTree(torch.nn.Module):

    def __init__(self, model_path: str, name="", link_list=None, device='cpu'):

        super().__init__()

        self.name = name
        self.link_list = link_list

        self.device = device
        self.model_type = model_path.split('.')[-1]
        self._model = URDFRobotModel(model_path=model_path, device=self.device)

        self._bodies = torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []

        # NOTE: making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # NOTE: joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self._model.links):
            # Initialize body object
            rigid_body_params = self._model.get_body_parameters(i, link)
            body = DifferentiableRigidBody(
                rigid_body_params=rigid_body_params, device=self.device
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
    
    def reset(self):
        '''Reset the robots FK after stateful update'''
        for body in self._bodies:
            body.reset()

    def update_base_pose(self, pose_vec):
        self._bodies[0].update_pose(pose_vec)

    def update_kinematic_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        """
        Updates the kinematic state of the robots in a stateful way
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
            torch.zeros((batch_size, 3), device=self.device),
            torch.zeros((batch_size, 3), device=self.device),
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
            return self.compute_forward_kinematics_all_links(q, link_list=[link_name])  # return SE3 matrix
        else:
            self.update_kinematic_state(q, qd)

            pose = self._bodies[self._name_to_idx_map[link_name]].pose
            pos = pose.translation
            rot = pose.get_quaternion()
            rot = q_convert_wxyz(rot)  # pose.get_quaternion() returns a quaternion xyzw
            return pos, rot  # return tuple of translation and rotation

    def compute_forward_kinematics_and_geometric_jacobian(
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

        lin_jac = torch.zeros([batch_size, 3, self._n_dofs], device=self.device)
        ang_jac = torch.zeros([batch_size, 3, self._n_dofs], device=self.device)
        # any joints larger than this joint, will have 0 in the jacobian
        # parent_name = self._urdf_model.get_name_of_parent_body(link_name)
        parent_joint_id = self._model.find_joint_of_body(link_name)

        # TODO - implement as a tensor cross product
        for i, idx in enumerate(self._controlled_joints):
            if (idx - 1) > parent_joint_id:
                continue
            pose = self._bodies[idx].pose
            axis_idx = self._bodies[idx].axis_idx
            p_i = pose.translation
            z_i = torch.index_select(pose.rotation, -1, axis_idx).squeeze(-1)
            lin_jac[:, :, i] = torch.cross(z_i, ee_pos - p_i)
            ang_jac[:, :, i] = z_i

        return ee_pos, ee_rot, lin_jac, ang_jac

    def compute_analytical_jacobian_all_links(
        self, q: torch.Tensor
    ):
        # Batch computation of analytical jacobian for all links
        if q.ndim == 1:
            q = q.unsqueeze(0)

        def surrogate_fn(x):
            # Surrogate function that sums over the batch size
            links_tensor = self.compute_forward_kinematics_all_links(x)
            pos = link_pos_from_link_tensor(links_tensor)
            quat = link_quat_from_link_tensor(links_tensor)
            return torch.cat((pos, quat), dim=-1).sum(0)

        links_jac = jacobian(surrogate_fn, q).movedim(-2, 0)
        return links_jac

    def compute_forward_kinematics_all_links(
        self, q: torch.Tensor, return_dict=False, link_list=None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stateless forward kinematics
        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: dict link frame

        """
        if q.ndim == 1:
            q = q.unsqueeze(0)

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

    def inverse_kinematics(
            self, H_target, link_name='ee_link',
            batch_size=1,
            max_iters=1000, lr=1e-2,
            se3_eps=1e-1,
            q0=None,
            q0_noise=torch.pi/8,
            eps_joint_lim=torch.pi/100,
            print_freq=50,
            debug=False
    ):
        """
        Solve IK using Adam optimizer
        Args:
            H_target: target SE3 matrix [batch_size x 4 x 4]
            num_iters: number of iterations
            se3_eps: threshold for stopping the iteration
            q0: initial guess for the joint angles

        Returns: joint angles

        """
        if H_target.ndim == 2:
            H_target = H_target.unsqueeze(0)

        # Sample configurations uniformly
        lower, upper, _, _ = self.get_joint_limit_array()
        lower += eps_joint_lim
        upper -= eps_joint_lim
        lower = to_torch(lower, device=self.device)
        upper = to_torch(upper, device=self.device)
        if q0 is None:
            q0 = torch.rand(batch_size, self._n_dofs, device=self.device)
            q0 = lower + q0 * (upper - lower)
        else:
            # add some noise to get diverse solutions
            q0 += torch.randn(batch_size, self._n_dofs, device=self.device) * q0_noise
            q0 = torch.clamp(q0, lower, upper)
            assert q0.shape[0] == batch_size
            assert q0.shape[1] == self._n_dofs

        # Optimize
        q = q0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([q], lr=lr)
        for i in range(max_iters):
            optimizer.zero_grad()
            # Check if all configurations respect the termination criteria
            idx_valid = self.ik_termination(
                q, H_target, link_name,
                lower, upper,
                se3_eps=se3_eps,
                debug=debug
            )
            if idx_valid.nelement() == batch_size:
                print(f'\nIK converged for all joint configurations in {i} iterations')
                break

            # TODO - optimize only the joint configurations that are not valid

            # loss and gradient step
            err_per_q = self.loss_fn_ik_per_q(
                q, H_target, link_name,
                w_se3=1.0,
                w_joint_limits=300.0, lower=lower, upper=upper,
                w_q_rest=1., q_rest=None,
                debug=debug
            )
            if (i == 0 or (i % print_freq) == 0) and print_freq != -1:
                print(f'\n---> Iter {i}/{max_iters}')
                print(f'Error mean, std: {err_per_q.mean():.3f}, {err_per_q.std():.3f}')
                print(f'idx_valid: {len(idx_valid)}/{batch_size}')

            # do not optimize configurations that are already valid
            err_per_q.sum().backward()
            optimizer.step()

        if i == max_iters - 1:
            print(f'\nIK did not converge for all joint configurations!')
            print(f'Error mean, std: {err_per_q.mean():.3f}, {err_per_q.std():.3f}')
            print(f'idx_valid: {len(idx_valid)}/{batch_size}')

        return q, idx_valid

    def loss_fn_ik_per_q(
            self,
            q, H_target, link_name,
            w_se3=1.0,
            w_joint_limits=1.0, lower=None, upper=None,
            w_q_rest=1., q_rest=None,
            debug=False
    ):
        # EE cost
        H = self.compute_forward_kinematics_all_links(q, link_list=[link_name]).squeeze(1)
        err_se3 = SE3_distance(H, H_target, w_pos=1., w_rot=1.)

        # Joint limit cost
        lower_mask = torch.where(q < lower, 1, 0)
        err_joint_limit_lower = ((lower - q).pow(2) * lower_mask).sum(-1)
        upper_mask = torch.where(q > upper, 1, 0)
        err_joint_limit_upper = ((upper - q).pow(2) * upper_mask).sum(-1)
        err_joint_limits = err_joint_limit_lower + err_joint_limit_upper

        # Rest configuration cost
        err_q_rest = 0.
        if q_rest is not None:
            err_q_rest = torch.linalg.norm(q - q_rest, dim=-1)

        # Total error
        if debug:
            print(f'w_se3 * err_se3                  : {w_se3 * err_se3}')
            print(f'w_joint_limits * err_joint_limits: {w_joint_limits * err_joint_limits}')
            print(f'w_q_rest * err_q_rest            : {w_q_rest * err_q_rest}')

        err = w_se3 * err_se3 + w_joint_limits * err_joint_limits + w_q_rest * err_q_rest
        return err

    def ik_termination(
            self,
            q, H_target, link_name,
            lower, upper,
            se3_eps=1e-1,
            debug=False
    ):
        # Check joint limits
        idx_valid_joints = torch.argwhere(torch.all(torch.logical_and(q >= lower, q <= upper), dim=-1)).squeeze()
        idx_valid_joints = torch.atleast_1d(idx_valid_joints)

        # Check SE3 distance
        H = self.compute_forward_kinematics_all_links(q, link_list=[link_name]).squeeze(1)
        err_se3 = SE3_distance(H, H_target, w_pos=1., w_rot=1.)
        idx_valid_se3 = torch.argwhere(err_se3 < se3_eps).squeeze()
        idx_valid_se3 = torch.atleast_1d(idx_valid_se3)

        if debug:
            print(f'#idx_valid_joints  : {len(idx_valid_joints) if idx_valid_joints.nelement()>0 else 0}')
            print(f'#idx_valid_se3     : {len(idx_valid_se3) if idx_valid_se3.nelement()>0 else 0}')

        idx_valid = torch_intersect_1d(idx_valid_joints, idx_valid_se3)

        return idx_valid

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
    
    def print_dynamics_info(self) -> None:
        """

        print the names of all links

        """
        for i in range(len(self._bodies)):
            print(self._bodies[i].name)
            print(self._bodies[i]._get_dynamics_parameters_values())

    def iterative_newton_euler(
        self, base_lin_acc: torch.Tensor, base_ang_acc: torch.Tensor
    ) -> None:
        r"""

        Args:
            base_lin_acc: linear acceleration of base (for fixed manipulators this is zero)
            base_ang_acc: angular acceleration of base (for fixed manipulators this is zero)

        """

        body = self._bodies[0]
        body.joint_acc = MotionVec(base_lin_acc, base_ang_acc)

        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = self._model.get_name_of_parent_body(body.name)

            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()
            new_acc = parent_body.acc.transform(inv_pose).add_motion_vec(body.joint_acc)

            # body velocity cross joint vel
            bv_cross_jv = body.vel.cross_motion_vec(body.joint_vel)
            body.acc = bv_cross_jv.add_motion_vec(new_acc)

        child_body = self._bodies[-1]

        # after recursion is done, we propagate forces back up (from endeffector link to base)
        for i in range(len(self._bodies) - 2, 0, -1):
            body = self._bodies[i]
            joint_pose = child_body.joint_pose

            # pose x children_force
            child_force = child_body.force.transform(joint_pose)

            IcAcc = body.multiply_inertia_with_motion_vec(body.acc)
            IcVel = body.multiply_inertia_with_motion_vec(body.vel)

            # body vel x IcVel
            tmp_force = IcVel.cross_motion_vec(body.vel)

            body.force = child_force.add_motion_vec(tmp_force).add_motion_vec(IcAcc)
            child_body = body

    def compute_inverse_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd_des: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            qdd_des: desired joint accelerations [batch_size x n_dofs]
            include_gravity: when False, we assume gravity compensation is already taken care off

        Returns: forces to achieve desired accelerations

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert qdd_des.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs
        assert qdd_des.shape[1] == self._n_dofs
        
        q = q.to(device=self.device)
        qd = qd.to(device=self.device)
        qdd_des = qdd_des.to(device=self.device)

        batch_size = qdd_des.shape[0]
        force = torch.zeros_like(qdd_des)

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # we set the acceleration of all controlled joints to the desired accelerations
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_acc(qdd_des[:, i].unsqueeze(1))

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size, device=self.device)

        # we propagate the base forces
        self.iterative_newton_euler(base_lin_acc, base_ang_acc)

        # we extract the relevant forces for all controlled joints
        for i in range(qdd_des.shape[1]):
            idx = self._controlled_joints[i]
            rot_axis = torch.zeros((batch_size, 3)).to(device=self.device)
            rot_axis[:, 2] = torch.ones(batch_size).to(device=self.device)
            force[:, i] += (
                self._bodies[idx].force.ang.unsqueeze(1) @ rot_axis.unsqueeze(2)
            ).squeeze()

        # we add forces to counteract damping
        damping_const = torch.zeros(1, self._n_dofs, device=self.device)
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
        force += damping_const.repeat(batch_size, 1) * qd

        return force

    def compute_non_linear_effects(
        self, q: torch.Tensor, qd: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Compute the non-linear effects (Coriolis, centrifugal, gravitational, and damping effects).

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        zero_qdd = q.new_zeros(q.shape)
        return self.compute_inverse_dynamics(q, qd, zero_qdd, include_gravity)

    def compute_lagrangian_inertia_matrix(
        self, q: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        assert q.shape[1] == self._n_dofs
        batch_size = q.shape[0]
        identity_tensor = torch.eye(q.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_qd = q.new_zeros(q.shape)
        zero_qdd = q.new_zeros(q.shape)
        if include_gravity:
            gravity_term = self.compute_inverse_dynamics(
                q, zero_qd, zero_qdd, include_gravity
            )
        else:
            gravity_term = q.new_zeros(q.shape)

        H = torch.stack(
            [
                (
                    self.compute_inverse_dynamics(
                        q, zero_qd, identity_tensor[:, :, j], include_gravity
                    )
                    - gravity_term
                )
                for j in range(self._n_dofs)
            ],
            dim=2,
        )
        return H
    
    def get_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ):
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: M, C, G, D

        """

        nle = self.compute_non_linear_effects(
            q=q, qd=qd, include_gravity=include_gravity
        )
        inertia_mat = self.compute_lagrangian_inertia_matrix(
            q=q, include_gravity=include_gravity
        )
        damping_const = torch.zeros(1, self._n_dofs, device=self.device)
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
        return inertia_mat, nle, damping_const

    def compute_forward_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        f: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""
        Computes next qdd by solving the Euler-Lagrange equation
        qdd = H^{-1} (F - Cv - G - damping_term)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """

        inertia_mat, nle, _ = self.get_dynamics(q, qd, include_gravity)
        # Solve H qdd = F - Cv - G - damping_term
        print(f, nle)
        print(inertia_mat)
        qdd = torch.linalg.solve(inertia_mat, f - nle)

        return qdd
