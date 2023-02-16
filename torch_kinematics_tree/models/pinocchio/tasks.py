import abc
from typing import Tuple, Union, Sequence
import pinocchio as pin
import numpy as np

from .utils import pin_log_se3

class Task(abc.ABC):
    r"""Abstract base class for kinematic tasks.

    Attributes:
        weight: Task weight :math:`\alpha \in [0, 1]` for additional low-pass
            filtering. Defaults to 1.0 (no filtering) for dead-beat control.
    """

    weight: float = 1.0

    @abc.abstractmethod
    def compute_task_dynamics(
        self, q
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the task dynamics matrix and vector.

        Those are the matrix :math:`J(q)` and vector :math:`\alpha e(q)` such
        that the task dynamics are:

        .. math::

            J(q) \Delta q = \alpha e(q)

        The Jacobian matrix is :math:`J(q) \in \mathbb{R}^{k \times n_v}`,
        with :math:`n_v` the dimension of the robot's tangent space and
        :math:`k` the dimension of the task. The error vector :math:`e(q) \in
        \mathbb{R}^k` is multiplied by the task weight :math:`\alpha \in [0,
        1]`. The weight is usually 1 for dead-beat control (*i.e.* converge as
        fast as possible), but it can also be lower for some extra low-pass
        filtering.

        Both :math:`J(q)` and :math:`(e)` depend on the q :math:`q`
        of the robot. The q displacement :math:`\\Delta q` is the
        output of inverse kinematics.

        Args:
            q: Robot q to read values from.

        Returns:
            Tuple :math:`(J, e)` of Jacobian matrix and error vector, both
            expressed in the body frame.
        """

    @abc.abstractmethod
    def compute_qp_objective(
        self, q
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective
        of the IK is:

        .. math::

            \| J \Delta q - \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H
            \Delta q + c^T q

        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weighs and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2. The q displacement :math:`\Delta
        q` is the output of inverse kinematics (we divide it by :math:`\Delta
        t` to get a commanded velocity).

        Args:
            q: Robot q to read values from.

        Returns:
            Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
            objective.
        """

    def __repr__(self):
        return f"Task(weight={self.weight})"



class BodyTask(Task):
    r"""Regulate the pose of a robot body in the world frame.
    """

    def __init__(
        self,
        body: str,
        position_cost: Union[float, Sequence[float]],
        orientation_cost: Union[float, Sequence[float]],
        lm_damping: float = 1e-6,
    ) -> None:
        r"""Define a new body task.
        """
        self.body = body
        self.cost = np.ones(6)
        self.lm_damping = lm_damping
        self.transform_target_to_world = None
        #
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(
        self, position_cost: Union[float, Sequence[float]]
    ) -> None:
        r"""Set a new cost for all 3D position coordinates.
        """
        if isinstance(position_cost, float):
            assert position_cost >= 0.0
        else:  # not isinstance(position_cost, float)
            assert all(cost >= 0.0 for cost in position_cost)
        self.cost[0:3] = position_cost

    def set_orientation_cost(
        self, orientation_cost: Union[float, Sequence[float]]
    ) -> None:
        r"""Set a new cost for all 3D orientation coordinates.
        """
        if isinstance(orientation_cost, float):
            assert orientation_cost >= 0.0
        else:  # not isinstance(orientation_cost, float)
            assert all(cost >= 0.0 for cost in orientation_cost)
        self.cost[3:6] = orientation_cost

    def set_target(
        self,
        transform_target_to_world: pin.SE3,
    ) -> None:
        """Set task target pose in the world frame.
        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(
        self, pin_model
    ) -> None:
        """Set task target pose from a robot configuration.
        """
        self.set_target(pin_model.get_transform_body_to_world(self.body))

    def compute_error_in_body(
        self, pin_model
    ) -> np.ndarray:
        r"""Compute the body twist error.
        """
        if self.transform_target_to_world is None:
            raise Exception("Target pose is not set.")
        transform_body_to_world = pin_model.get_transform_body_to_world(
            self.body
        )
        error_in_body = pin_log_se3(
            self.transform_target_to_world,
            transform_body_to_world,
        )
        return error_in_body

    def compute_task_dynamics(
        self, pin_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the task dynamics matrix and vector.
        """
        jacobian_in_body = pin_model.get_body_jacobian(self.body)

        # TODO: fix sign of error and box minus
        if self.transform_target_to_world is None:
            raise Exception("Target pose is not set.")
        transform_body_to_world = pin_model.get_transform_body_to_world(
            self.body
        )
        transform_body_to_target = (
            self.transform_target_to_world.inverse() * transform_body_to_world
        )
        J = pin.Jlog6(transform_body_to_target) @ jacobian_in_body

        error_in_body = self.compute_error_in_body(pin_model)
        return J, self.weight * error_in_body

    def compute_qp_objective(
        self, pin_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.
        """
        jacobian, error = self.compute_task_dynamics(pin_model)
        weight = np.diag(self.cost)  # [cost] * [twist]^{-1}
        weighted_jacobian = weight @ jacobian  # [cost]
        weighted_error = weight @ error  # [cost]
        mu = self.lm_damping * weighted_error @ weighted_error  # [cost]^2
        eye_tg = pin_model.tangent.eye
        # Our Levenberg-Marquardt damping `mu * eye_tg` is isotropic in the
        # robot's tangent space. If it helps we can add a tangent-space scaling
        # to damp the floating base differently from joint angular velocities.
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        return (
            f"BodyTask({self.body}, "
            f"weight={self.weight}, "
            f"orientation_cost={self.cost[3:6]}, "
            f"position_cost={self.cost[0:3]}, "
            f"target={self.transform_target_to_world})"
        )
