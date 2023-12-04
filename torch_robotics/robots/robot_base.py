import abc
from abc import ABC
from copy import copy
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import einops
import torch
import yaml
from filelock import FileLock
from urdf_parser_py.urdf import URDF, Joint, Link, Visual, Collision, Pose, Sphere

import torchkin
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_convert_to_xyzw, q_to_euler, \
    rotation_matrix_to_q
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor, link_rot_from_link_tensor, link_quat_from_link_tensor)
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch
from torch_robotics.trajectory.utils import finite_difference_vector


def write_robot_urdf_file(robot_urdf_file, xmlstr):
    robot_urdf_file_posix = robot_urdf_file.as_posix()
    robot_file_posix_lockfile = robot_urdf_file_posix + ".lock"
    lock = FileLock(robot_file_posix_lockfile, timeout=120)
    lock.acquire()
    try:
        with open(str(robot_urdf_file), "w") as f:
            f.write(xmlstr)
    finally:
        lock.release()


def modidy_robot_urdf_collision_model(
        urdf_robot_file,
        collision_spheres_file_path):
    # load collision spheres file
    coll_yml = collision_spheres_file_path
    with open(coll_yml) as file:
        coll_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_urdf = URDF.from_xml_file(urdf_robot_file)

    link_collision_names = []
    link_collision_margins = []
    for link_name, spheres_l in coll_params.items():
        for i, sphere in enumerate(spheres_l):
            link_collision = f'{link_name}_{i}'
            joint = Joint(
                name=f'joint_{link_name}_sphere_{i}',
                parent=f'{link_name}',
                child=link_collision,
                joint_type='fixed',
                origin=Pose(xyz=to_numpy(sphere[:3]))
            )
            robot_urdf.add_joint(joint)

            link = Link(
                name=link_collision,
                # visual=Visual(Sphere(sphere[-1])),  # add collision sphere to model
                origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
            )
            robot_urdf.add_link(link)

            link_collision_names.append(link_collision)
            link_collision_margins.append(sphere[-1])

    # replace the robots file
    robot_urdf_file = Path(str(urdf_robot_file).replace('.urdf', '_collision_model.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")

    # Write the file with a lock
    write_robot_urdf_file(robot_urdf_file, xmlstr)

    return robot_urdf_file, link_collision_names, link_collision_margins


def modidy_robot_urdf_grasped_object(urdf_robot_file, grasped_object, parent_link):
    robot_urdf = URDF.from_xml_file(urdf_robot_file)

    # Add the grasped object visual and collision
    link_grasped_object = f'link_{grasped_object.name}'
    joint = Joint(
        name=f'joint_fixed_{grasped_object.name}',
        parent=parent_link,
        child=link_grasped_object,
        joint_type='fixed',
        origin=Pose(xyz=to_numpy(grasped_object.pos.squeeze()),
                    rpy=to_numpy(q_to_euler(rotation_matrix_to_q(grasped_object.ori)).squeeze())
                    )
    )
    robot_urdf.add_joint(joint)

    geometry_grasped_object = grasped_object.geometry_urdf
    link = Link(
        name=link_grasped_object,
        visual=Visual(geometry_grasped_object),
        # inertial=None,
        collision=Collision(geometry_grasped_object),
        origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
    )
    robot_urdf.add_link(link)

    # Create fixed joints and links for the grasped object collision points
    link_collision_names = []
    for i, point_collision in enumerate(grasped_object.points_for_collision):
        link_collision = f'link_{grasped_object.name}_point_{i}'
        joint = Joint(
            name=f'joint_fixed_{grasped_object.name}_point_{i}',
            parent=link_grasped_object,
            child=link_collision,
            joint_type='fixed',
            origin=Pose(xyz=to_numpy(point_collision))
        )
        robot_urdf.add_joint(joint)

        link = Link(
            name=link_collision,
            origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
        )
        robot_urdf.add_link(link)

        link_collision_names.append(link_collision)

    # replace the robots file
    robot_file = Path(str(urdf_robot_file).replace('.urdf', '_grasped_object.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")

    # Write the file with a lock
    write_robot_urdf_file(robot_file, xmlstr)

    return robot_file, link_collision_names


class RobotBase(ABC):
    link_name_ee = None

    def __init__(
            self,
            urdf_robot_file,
            collision_spheres_file_path,
            link_name_ee=None,
            gripper_q_dim=0,
            grasped_object=None,
            dt=1.0,
            task_space_dim=3,
            tensor_args=None,
            **kwargs
    ):
        self.name = self.__class__.__name__
        self.tensor_args = tensor_args

        assert link_name_ee is not None, 'link_name_ee must be defined'
        self.link_name_ee = link_name_ee
        self.gripper_q_dim = gripper_q_dim
        self.grasped_object = grasped_object

        self.dt = dt  # time interval to compute velocities and accelerations from positions via finite difference

        # If the task space is 2D (point mass or plannar robot), then the z coordinate is set to 0
        self.task_space_dim = task_space_dim

        ################################################################################################
        # Robot collision model (links and margins) for object collision avoidance
        self.link_object_collision_names = []
        self.link_object_collision_margins = []

        # Modify the urdf to append the link and collision points of the grasped object
        if grasped_object is not None:
            urdf_robot_file, link_collision_names = modidy_robot_urdf_grasped_object(
                urdf_robot_file, grasped_object, 'panda_hand')
            self.link_object_collision_names.extend(link_collision_names)
            self.link_object_collision_margins.extend([grasped_object.object_collision_margin] * len(link_collision_names))

        # The raw version of the original urdf file with the grasped object is used
        # for collision checking and visualization
        try:
            self.urdf_robot_file_raw = copy(urdf_robot_file).as_posix()
        except AttributeError:
            self.urdf_robot_file_raw = copy(urdf_robot_file)

        # Modify the urdf to append links of the collision model
        urdf_robot_file, link_collision_names, link_collision_margins = modidy_robot_urdf_collision_model(
            urdf_robot_file, collision_spheres_file_path)
        self.link_object_collision_names.extend(link_collision_names)
        self.link_object_collision_margins.extend(link_collision_margins)

        assert len(self.link_object_collision_names) == len(self.link_object_collision_margins)

        self.link_object_collision_margins = to_torch(self.link_object_collision_margins, **tensor_args)

        self.urdf_robot_file = urdf_robot_file.as_posix()

        ################################################################################################
        # Configuration space limits
        # Lock the urdf robot file because of multiprocessing
        urdf_robot_file_posix = self.urdf_robot_file
        urdf_robot_file_posix_lockfile = urdf_robot_file_posix + ".lock"
        lock = FileLock(urdf_robot_file_posix_lockfile, timeout=120)
        lock.acquire()
        try:
            robot_urdf = URDF.from_xml_file(self.urdf_robot_file)
        finally:
            lock.release()

        q_limits_lower = []
        q_limits_upper = []
        for joint in robot_urdf.joints:
            if joint.joint_type != 'fixed':
                q_limits_lower.append(joint.limit.lower)
                q_limits_upper.append(joint.limit.upper)

        self.q_dim = len(q_limits_lower)
        self.gripper_q_dim = gripper_q_dim
        self.arm_q_dim = self.q_dim - self.gripper_q_dim
        self.q_min = to_torch(q_limits_lower, **tensor_args)
        self.q_max = to_torch(q_limits_upper, **tensor_args)
        self.q_min_np = to_numpy(self.q_min)
        self.q_max_np = to_numpy(self.q_max)
        self.q_distribution = torch.distributions.uniform.Uniform(self.q_min, self.q_max)

        ################################################################################################
        # Torchkin robot forward kinematics functions
        # Lock the urdf robot file because of multiprocessing
        urdf_robot_file_posix = self.urdf_robot_file
        urdf_robot_file_posix_lockfile = urdf_robot_file_posix + ".lock"
        lock = FileLock(urdf_robot_file_posix_lockfile, timeout=120)
        lock.acquire()
        try:
            self.robot_torchkin = torchkin.Robot.from_urdf_file(self.urdf_robot_file, **tensor_args)
        finally:
            lock.release()

        print('-----------------------------------')
        print(f'Torchkin robot: {self.robot_torchkin.name}')
        print(f'Num links: {len(self.robot_torchkin.get_links())}')
        print(f'DOF: {self.robot_torchkin.dof}\n')
        print('-----------------------------------')

        # kinematic functions for object collision
        fk_object_collision, jfk_b_object_collision, jfk_s_object_collision = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=self.link_object_collision_names)
        self.fk_object_collision = fk_object_collision
        self.jfk_b_object_collision = jfk_b_object_collision
        self.jfk_s_object_collision = jfk_s_object_collision

        # kinematic functions for end-effector link
        fk_ee, jfk_b_ee, jfk_s_ee = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=[self.link_name_ee])
        self.fk_ee = fk_ee
        self.jfk_b_ee = jfk_b_ee
        self.jfk_s_ee = jfk_s_ee

        # kinematic functions for all links
        fk_all, jfk_b_all, jfk_s_all = torchkin.get_forward_kinematics_fns(robot=self.robot_torchkin)
        self.fk_all = fk_all
        self.jfk_b_all = jfk_b_all
        self.jfk_s_all = jfk_s_all

        ################################################################################################
        # Self collision field
        self.df_collision_self = None

        ################################################################################################
        # Grasped object
        self.grasped_object = grasped_object

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    def get_position(self, x):
        return x[..., :self.q_dim]

    def get_velocity(self, x):
        vel = x[..., self.q_dim:2 * self.q_dim]
        # If there is no velocity in the state, then compute it via finite difference
        if x.nelement() != 0 and vel.nelement() == 0:
            vel = finite_difference_vector(x, dt=self.dt, method='central')
            return vel
        return vel

    def get_acceleration(self, x):
        acc = x[..., 2 * self.q_dim:3 * self.q_dim]
        # If there is no acceleration in the state, then compute it via finite difference
        if x.nelement() != 0 and acc.nelement() == 0:
            vel = self.get_velocity(x)
            acc = finite_difference_vector(vel, dt=self.dt, method='central')
            return acc
        return acc

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError

    def get_EE_pose(self, q, flatten_pos_quat=False, quat_xyzw=False):
        _q = q
        if _q.ndim == 1:
            _q = _q.unsqueeze(0)
        if flatten_pos_quat:
            orientation_quat_wxyz = self.get_EE_orientation(_q, rotation_matrix=False)
            orientation_quat = orientation_quat_wxyz
            if quat_xyzw:
                orientation_quat = q_convert_to_xyzw(orientation_quat_wxyz)
            return torch.cat((self.get_EE_position(_q), orientation_quat), dim=-1)
        else:
            pose = self.fk_ee(_q)[0]
            return pose

    def get_EE_position(self, q):
        ee_pose = self.get_EE_pose(q)
        return link_pos_from_link_tensor(ee_pose)

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        if rotation_matrix:
            return link_rot_from_link_tensor(ee_pose)
        else:
            return link_quat_from_link_tensor(ee_pose)

    def fk_map_collision(self, q, **kwargs):
        _q = q
        if _q.ndim == 1:
            _q = _q.unsqueeze(0)  # add batch dimension
        task_space_positions = self.fk_map_collision_impl(_q, **kwargs)
        # Filter the positions from FK to the dimensions of the environment
        # Some environments are defined in 2D, while the robot FK is always defined in 3D
        task_space_positions = task_space_positions[..., :self.task_space_dim]
        return task_space_positions

    def fk_map_collision_impl(self, q, **kwargs):
        # q: (..., q_dim)
        # return: (..., links_collision_positions, 3)
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        link_poses = self.fk_object_collision(q)
        links_poses_th = torch.stack(link_poses).transpose(0, 1)

        if len(q_orig_shape) == 3:
            links_poses_th = einops.rearrange(links_poses_th, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        link_positions_th = link_pos_from_link_tensor(links_poses_th)  # (batch horizon), taskspaces, x_dim

        return link_positions_th
