from pathlib import Path
from typing import Optional, List
from xml.dom import minidom

import numpy as np
import yaml
from urdf_parser_py.urdf import URDF, Joint, Link, Visual, Collision, Box, Pose

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_euler
from torch_robotics.torch_kinematics_tree.models.robot_tree import DifferentiableTree
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from xml.etree import ElementTree as ET

from torch_robotics.torch_utils.torch_utils import to_numpy


class DifferentiableKUKAiiwa(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'kuka_iiwa' / 'urdf' / 'iiwa7.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


def modidy_robot_urdf_collision_model(robot_file, collision_spheres_file_path='panda/panda_sphere_config.yaml'):
    # load collision spheres file
    coll_yml = (get_configs_path() / collision_spheres_file_path).as_posix()
    with open(coll_yml) as file:
        coll_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_urdf = URDF.from_xml_file(robot_file)

    link_collision_names = []
    link_collision_margins = []
    for link_name, spheres_l in coll_params.items():
        for i, sphere in enumerate(spheres_l):
            joint = Joint(
                name=f'joint_{link_name}_sphere_{i}',
                parent=f'{link_name}',
                child=f'{link_name}_{i}',
                joint_type='fixed',
                origin=Pose(xyz=to_numpy(sphere[:3]))
            )
            robot_urdf.add_joint(joint)

            link_collision = f'{link_name}_{i}'
            link = Link(
                name=link_collision,
                origin=Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
            )
            robot_urdf.add_link(link)

            link_collision_names.append(link_collision)
            link_collision_margins.append(sphere[-1])

    # replace the robots file
    robot_file = Path(str(robot_file).replace('.urdf', '_collision_model.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file, link_collision_names, link_collision_margins


def modidy_robot_urdf_grasped_object(robot_file, grasped_object, parent_link):
    robot_urdf = URDF.from_xml_file(robot_file)

    # Add the grasped object visual and collision
    link_grasped_object = f'link_{grasped_object.name}'
    joint = Joint(
        name=f'joint_fixed_{grasped_object.name}',
        parent=parent_link,
        child=link_grasped_object,
        joint_type='fixed',
        origin=Pose(xyz=to_numpy(grasped_object.pos.squeeze()),
                    rpy=to_numpy(q_to_euler(grasped_object.ori).squeeze())
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
            parent=f'{grasped_object.reference_frame}',
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
    robot_file = Path(str(robot_file).replace('.urdf', '_grasped_object.urdf'))
    xmlstr = minidom.parseString(ET.tostring(robot_urdf.to_xml())).toprettyxml(indent="   ")
    with open(str(robot_file), "w") as f:
        f.write(xmlstr)

    return robot_file, link_collision_names


class DifferentiableFrankaPanda(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, gripper=False, device='cpu', grasped_object=None):
        if gripper:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand.urdf'
        else:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand_no_gripper.urdf'

        self.link_collision_names = []
        self.link_collision_margins = []
        # Modify the urdf to append links of the collision model
        robot_file, link_collision_names, link_collision_margins = modidy_robot_urdf_collision_model(
            robot_file, collision_spheres_file_path='panda/panda_sphere_config.yaml')
        self.link_collision_names.extend(link_collision_names)
        self.link_collision_margins.extend(link_collision_margins)

        # Modify the urdf to append the link and collision points of the grasped object
        if grasped_object is not None:
            robot_file, link_collision_names = modidy_robot_urdf_grasped_object(
                robot_file, grasped_object, 'panda_hand')
            self.link_collision_names.extend(link_collision_names)
            self.link_collision_margins.extend([grasped_object.object_collision_margin] * len(link_collision_names))

        self.model_path = robot_file.as_posix()
        self.name = "differentiable_franka_panda"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableUR10(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, attach_gripper=False, device='cpu'):
        robot_path = get_robot_path()
        if attach_gripper:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10_suction.urdf'
        else:
            robot_file = robot_path / 'ur10' / 'urdf' / 'ur10.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_ur10"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableHabitatStretch(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_path = get_robot_path()
        robot_file = robot_path / 'habitat_stretch' / 'urdf' / 'hab_stretch.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_stretch"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHolo(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableTiagoDualHoloMove(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'tiago_dual_description' / 'tiago_dual_holobase_minimal_holonomic.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_tiago_dual_holo_move"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)

    def get_link_names(self):  # pop those hacky frames for moving base
        return super().get_link_names()[3:]


class DifferentiableShadowHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'shadow_hand' / 'shadow_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_shadow_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableAllegroHand(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'allegro_hand' / 'allegro_hand.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_allegro_hand"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class Differentiable2LinkPlanar(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'planar_manipulators' / 'urdf' / '2_link_planar.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_2_link_planar"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)
