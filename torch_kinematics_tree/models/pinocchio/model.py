import numpy as np
import pinocchio as pin
import qpsolvers
from .bounded_tangent import BoundedTangent, VectorSpace
from .utils import get_root_joint_dim, load_pinocchio_robot_description

from torch_kinematics_tree.utils.files import get_robot_path


class PinocchioModel(object):
    """Kinodynamic model based on Pinocchio."""
    def __init__(
        self,
        model_path,
        q=None,
    ):

        self.robot = load_pinocchio_robot_description(model_path, mesh_dirs=get_robot_path().as_posix())
        self.model = self.robot.model
        self.data = self.robot.data
        if not hasattr(self.model, "tangent"):
            self.model.tangent = VectorSpace(self.model.nv)
        if not hasattr(self.model, "bounded_tangent"):
            self.model.bounded_tangent = BoundedTangent(self.model)
        if q is None:
            q = pin.neutral(self.model)
        self.q = q
        self.tangent = self.model.tangent
        self.bounded_tangent = self.model.bounded_tangent

        self.update()

    def update(self) -> None:
        """Run forward kinematics from the configuration."""
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def check_limits(self, tol: float = 1e-6) -> None:
        """Check that the current configuration is within limits.
        """
        q_max = self.model.upperPositionLimit
        q_min = self.model.lowerPositionLimit
        root_nq, _ = get_root_joint_dim(self.model)
        for i in range(root_nq, self.model.nq):
            if q_max[i] <= q_min[i] + tol:  # no limit
                continue
            if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
                raise Exception( 
                    f"Joint {i} is out of limits: {self.q[i]} not in [{q_min[i]}, {q_max[i]}]")

    def get_body_jacobian(self, body: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of the body velocity.

        This matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            body: Body frame name, typically the link name from the URDF.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the body twist.

        When the robot model includes a floating base
        (pin.JointModelFreeFlyer), the configuration vector :math:`q` consists
        of:

        - ``q[0:3]``: position in [m] of the floating base in the inertial
          frame, formatted as :math:`[p_x, p_y, p_z]`.
        - ``q[3:7]``: unit quaternion for the orientation of the floating base
          in the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
        - ``q[7:]``: joint angles in [rad].
        """
        if not self.model.existBodyName(body):
            raise Exception(f"body {body} does not exist")
        body_id = self.model.getBodyId(body)
        J: np.ndarray = pin.getFrameJacobian(
            self.model, self.data, body_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def get_transform_body_to_world(self, body: str) -> pin.SE3:
        """Get the pose of a body frame in the current configuration.
        """
        body_id = self.model.getBodyId(body)
        try:
            return self.data.oMf[body_id].copy()
        except IndexError as index_error:
            raise KeyError(
                f'Body "{body}" not found in robot model'
            ) from index_error

    def compute_velocity_limits(self, dt: float, config_limit_gain: float = 0.5) -> np.ndarray:
        assert 0.0 < config_limit_gain <= 1.0
        bounded_tangent = self.bounded_tangent
        if bounded_tangent.velocity_limit is None:
            return None, None

        # Velocity limits from configuration bounds
        Delta_q_max = bounded_tangent.project(
            pin.difference(
                self.model,
                self.q,
                self.model.upperPositionLimit,
            )
        )
        Delta_q_min = bounded_tangent.project(
            pin.difference(
                self.model,
                self.q,
                self.model.lowerPositionLimit,
            )
        )

        # Intersect with velocity limits from URDF
        v_max = np.minimum(
            bounded_tangent.velocity_limit,
            config_limit_gain * Delta_q_max / dt,
        )
        v_min = np.maximum(
            -bounded_tangent.velocity_limit,
            config_limit_gain * Delta_q_min / dt,
        )

        return v_max, v_min
    
    def build_ik(self, tasks, dt, damping=0.0, config_limit_gain=0.5):
        r"""Build quadratic program from current configuration and tasks.

            This quadratic program is, in standard form:

            .. math::

                \begin{split}\begin{array}{ll}
                    \underset{\Delta q}{\mbox{minimize}} &
                        \frac{1}{2} {\Delta q}^T H {\Delta q} + c^T {\Delta q} \\
                    \mbox{subject to}
                        & G {\Delta q} \leq h
                \end{array}\end{split}

            where :math:`\Delta q` is a vector of joint displacements corresponding to
            the joint velocities :math:`v = {\Delta q} / {\mathrm{d}t}`.
        """
        # self.check_limits()

        # compute QP objectives
        P = damping * self.tangent.eye
        c = self.tangent.zeros.copy()
        for task in tasks:
            H_task, c_task = task.compute_qp_objective(self)
            P += H_task
            c += c_task

        # compute QP bounds
        v_max, v_min = self.compute_velocity_limits(dt, config_limit_gain=config_limit_gain)
        bounded_tangent = self.model.bounded_tangent
        bounded_proj = bounded_tangent.projection_matrix
        G = np.vstack([bounded_proj, -bounded_proj])
        h = np.hstack([dt * v_max, -dt * v_min])
        return qpsolvers.Problem(P, c, G, h)
    
    def solve_ik(self, tasks, dt, solver='quadprog', max_iters=1000, eps=1e-3, damping=0.01, config_limit_gain=0.5, **kwargs):
        i = 0
        while i < max_iters:
            problem = self.build_ik(tasks, dt, damping=damping, config_limit_gain=config_limit_gain)
            result = qpsolvers.solve_problem(problem, solver=solver, verbose=False)
            Delta_q = result.x
            if Delta_q is None:
                print("QP solver failed! Returning last configuration.")
                return self.q
            v = Delta_q / dt
            self.q = self.integrate_inplace(v, dt)
            if np.all(np.abs(v) < eps):
                return self.q
            i += 1
        print("IK solver failed to converge!")

    def integrate(self, velocity, dt) -> np.ndarray:
        return pin.integrate(self.model, self.q, velocity * dt)

    def integrate_inplace(self, velocity, dt) -> None:
        self.q = pin.integrate(self.model, self.q, velocity * dt)
        self.update()
