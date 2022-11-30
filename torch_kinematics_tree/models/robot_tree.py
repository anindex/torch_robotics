from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
import numpy as np
from torch_kinematics_tree.models.rigid_body import DifferentiableRigidBody
from torch_kinematics_tree.models.utils import URDFRobotModel, MJCFRobotModel
from torch_kinematics_tree.geometrics.spatial_vector import MotionVec


def tensor_check(function):
    """
    A decorator for checking the device of input tensors
    """

    @dataclass
    class BatchInfo:
        shape: torch.Size = torch.Size([])
        init: bool = False

    def preprocess(arg, obj, batch_info):
        if type(arg) is torch.Tensor:
            # Check dimensions & convert to 2-dim tensors
            assert arg.ndim in [1, 2], f"Input tensors must have ndim of 1 or 2."

            if batch_info.init:
                assert (
                    batch_info.shape == arg.shape[:-1]
                ), "Batch size mismatch between input tensors."
            else:
                batch_info.init = True
                batch_info.shape = arg.shape[:-1]

            if len(batch_info.shape) == 0:
                return arg.unsqueeze(0)

        return arg

    def postprocess(arg, batch_info):
        if type(arg) is torch.Tensor and batch_info.init and len(batch_info.shape) == 0:
            return arg[0, ...]

        return arg

    def wrapper(self, *args, **kwargs):
        batch_info = BatchInfo()

        # Parse input
        processed_args = [preprocess(arg, self, batch_info) for arg in args]
        processed_kwargs = {
            key: preprocess(kwargs[key], self, batch_info) for key in kwargs
        }

        # Perform function
        ret = function(self, *processed_args, **processed_kwargs)

        # Parse output
        if type(ret) is torch.Tensor:
            return postprocess(ret, batch_info)
        elif type(ret) is tuple:
            return tuple([postprocess(r, batch_info) for r in ret])
        else:
            return ret

    return wrapper


def convert_link_dict_to_tensor(link_dict, link_list):
    return torch.stack([link_dict[name].get_transform_matrix() for name in link_list], dim=1)


class DifferentiableTree(torch.nn.Module):

    def __init__(self, model_path: str, name="", link_list=None, device='cpu'):

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

    @tensor_check
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

    @tensor_check
    def compute_forward_kinematics_all_links(
        self, q: torch.Tensor, return_dict=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @tensor_check
    def compute_forward_kinematics_link_list(
        self, q: torch.Tensor, return_dict=False, link_list=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @tensor_check
    def compute_forward_kinematics(
        self, q: torch.Tensor, link_name: str, state_less: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        assert q.ndim == 2

        if state_less:
            return self.compute_forward_kinematics_all_links(q)[link_name]
        else:
            qd = torch.zeros_like(q)
            self.update_kinematic_state(q, qd)

            pose = self._bodies[self._name_to_idx_map[link_name]].pose
            pos = pose.translation
            rot = pose.get_quaternion()
            return pos, rot

    @tensor_check
    def compute_endeffector_jacobian(
        self, q: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]

        Returns: linear and angular jacobian

        """
        assert len(q.shape) == 2
        batch_size = q.shape[0]
        self.compute_forward_kinematics(q, link_name)

        e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        p_e = e_pose.translation

        lin_jac, ang_jac = (
            torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
            torch.zeros([batch_size, 3, self._n_dofs], device=self._device),
        )

        joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id

        while link_name != self._bodies[0].name:
            if joint_id in self._controlled_joints:
                i = self._controlled_joints.index(joint_id)
                idx = joint_id

                pose = self._bodies[idx].pose
                axis = self._bodies[idx].joint_axis
                p_i = pose.translation
                z_i = pose.rotation @ axis.squeeze()
                lin_jac[:, :, i] = torch.cross(z_i, p_e - p_i, dim=-1)
                ang_jac[:, :, i] = z_i
            link_name = self._model.get_name_of_parent_body(link_name)
            joint_id = self._bodies[self._name_to_idx_map[link_name]].joint_id

        return lin_jac, ang_jac

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
