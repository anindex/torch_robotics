from typing import Tuple, Optional, Union
import pinocchio as pin
import numpy as np

PinocchioJoint = Union[
    pin.JointModelRX,
    pin.JointModelRY,
    pin.JointModelRZ,
    pin.JointModelPX,
    pin.JointModelPY,
    pin.JointModelPZ,
    pin.JointModelFreeFlyer,
    pin.JointModelSpherical,
    pin.JointModelSphericalZYX,
    pin.JointModelPlanar,
    pin.JointModelTranslation,
]


def load_pinocchio_robot_description(
    urdf_file: str,
    mesh_dirs: Optional[str] = None,
    root_joint: Optional[PinocchioJoint] = None, 
) -> pin.RobotWrapper:
    """
    Load a robot description in Pinocchio.
    """
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_file,
        package_dirs=mesh_dirs,
        root_joint=root_joint,
    )
    return robot


def get_pin_configuration(robot: pin.Model, **kwargs) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.

    Args:
        robot: Robot model.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.
    """
    q = pin.neutral(robot.model)
    for joint_name, joint_value in kwargs.items():
        joint_id = robot.model.getJointId(joint_name)
        joint = robot.model.joints[joint_id]
        q[joint.idx_q] = joint_value
    return q


def get_root_joint_dim(model: pin.Model) -> Tuple[int, int]:
    """Count configuration and tangent dimensions of the root joint, if any.

    Args:
        model: Robot model.

    Returns:
        nq: Number of configuration dimensions.
        nv: Number of tangent dimensions.
    """
    if model.existJointName("root_joint"):
        root_joint_id = model.getJointId("root_joint")
        root_joint = model.joints[root_joint_id]
        return root_joint.nq, root_joint.nv
    return 0, 0


def pin_log_se3(Y: pin.SE3, X: pin.SE3) -> np.ndarray:
    r"""Compute the right minus :math:`Y \ominus X`.

    The right minus operator is defined by:

    .. math::

        Y \ominus X = \log(X^{-1} \cdot Y)
    """
    body_twist: np.ndarray = pin.log(X.actInv(Y)).vector
    return body_twist