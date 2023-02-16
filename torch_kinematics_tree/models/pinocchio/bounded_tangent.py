from typing import List, Optional

import numpy as np
import pinocchio as pin


class VectorSpace:
    """Wrapper to refer to a vector space and its characteristic matrices."""

    __dim: int
    __eye: np.ndarray
    __ones: np.ndarray
    __zeros: np.ndarray

    def __init__(self, dim: int):
        """Create new vector space description.

        Args:
            dim: Dimension.
        """
        eye = np.eye(dim)
        ones = np.ones(dim)
        zeros = np.zeros(dim)
        eye.setflags(write=False)
        ones.setflags(write=False)
        zeros.setflags(write=False)
        self.__dim = dim
        self.__eye = eye
        self.__ones = ones
        self.__zeros = zeros

    @property
    def dim(self) -> int:
        """Dimension of the vector space."""
        return self.__dim

    @property
    def eye(self) -> np.ndarray:
        """Identity matrix from and to the vector space."""
        return self.__eye

    @property
    def ones(self) -> np.ndarray:
        """Vector full of ones, dimension of the space."""
        return self.__ones

    @property
    def zeros(self) -> np.ndarray:
        """Zero vector of the space."""
        return self.__zeros



class BoundedTangent(VectorSpace):
    """Subspace of the tangent space restricted to bounded joints.

    Attributes:
        nv: Dimension of the full tangent space.
    """

    indices: np.ndarray
    joints: list
    nv: int
    projection_matrix: Optional[np.ndarray]
    velocity_limit: Optional[np.ndarray]

    def __init__(self, model: pin.Model):
        """Bounded joints in a robot model.

        Args:
            model: robot model.

        Returns:
            List of bounded joints.
        """
        has_configuration_limit = np.logical_and(
            model.upperPositionLimit < 1e20,
            model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
        )

        joints = [
            joint
            for joint in model.joints
            if joint.idx_q >= 0
            and has_configuration_limit[
                slice(joint.idx_q, joint.idx_q + joint.nq)
            ].all()
        ]

        index_list: List[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        indices = np.array(index_list)
        indices.setflags(write=False)

        dim = len(indices)
        super().__init__(dim)
        projection_matrix = np.eye(model.nv)[indices] if dim > 0 else None

        self.indices = indices
        self.joints = joints
        self.nv = model.nv
        self.projection_matrix = projection_matrix
        self.velocity_limit = (
            model.velocityLimit[indices] if len(joints) > 0 else None
        )

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project a tangent vector to the bounded tangent subspace.

        Args:
            v: Vector from the original space.
        """
        assert v.shape == (self.nv,), "Dimension mismatch"
        return v[self.indices]
