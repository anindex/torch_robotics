from typing import Optional, List
from torch_robotics.torch_kinematics_tree.models.robot_tree import DifferentiableTree
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path


class DifferentiableKUKAiiwa(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = get_robot_path() / 'kuka_iiwa' / 'urdf' / 'iiwa7.urdf'
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


class DifferentiableFrankaPanda(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, gripper=False, device='cpu'):
        if gripper:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_hand.urdf'
        else:
            robot_file = get_robot_path() / 'franka_description' / 'robots' / 'panda_arm_no_gripper.urdf'
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
