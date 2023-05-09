from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BodyCore(ABC):
    def __init__(
        self,
        base_position: Union[np.ndarray, list],
        base_orientation: Union[np.ndarray, list],
        scale: float = 1.0,
        fixed_base: bool = True,
    ) -> None:
        # Store initial position and orientation for resets
        self.init_base_position = base_position
        self.init_base_orientation = base_orientation
        self.fixed_base = fixed_base

        # Stick to base_position/_orientation for conformity with Robot definitions
        self._base_position = base_position
        self._base_orientation = base_orientation

        self.scale = scale

        self.id = None  # To be filled by Bullet client

        self.visual_id = None
        self.collision_id = None

    @property
    def base_orientation(self):
        return self._base_orientation

    @base_orientation.setter
    def base_orientation(self, values):
        self._base_orientation = values
        if hasattr(self, "client_id"):
            self.client_id.resetBasePositionAndOrientation(
                self.id, self.base_position, self.base_orientation
            )

    @property
    def base_position(self):
        return np.asarray(self._base_position)

    @base_position.setter
    def base_position(self, values):
        self._base_position = values
        if hasattr(self, "client_id"):
            self.client_id.resetBasePositionAndOrientation(
                self.id, self._base_position, self.base_orientation
            )

    def reset(self, **kwargs):
        self.base_orientation = self.init_base_orientation
        self.base_position = self.init_base_position

    @abstractmethod
    def load2client(self, client_id):
        raise NotImplementedError(
            "The method load2client() is not implemented in the abstract class."
        )


class DynamicBodyCore(BodyCore):
    def __init__(
        self,
        base_position: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        base_orientation: Union[np.ndarray, list] = [0.0, 0.0, 0.0, 1.0],
        base_linear_velocity: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        base_angular_velocity: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        scale: float = 1.0,
        fixed_base: bool = True,
    ) -> None:
        super(DynamicBodyCore, self).__init__(
            base_position=base_position,
            base_orientation=base_orientation,
            scale=scale,
            fixed_base=fixed_base,
        )
        # Store initial velocities
        self.init_base_linear_velocity = base_linear_velocity
        self.init_base_angular_velocity = base_angular_velocity

        # Stick to base_velocities for conformity with Robot definitions
        self._base_linear_velocity = base_linear_velocity
        self._base_angular_velocity = base_angular_velocity

    @property
    def base_linear_velocity(self):
        return self._base_linear_velocity

    @base_linear_velocity.setter
    def base_linear_velocity(self, values):
        self._base_linear_velocity = values
        if hasattr(self, "client_id"):
            self.client_id.resetBaseVelocity(
                self.id, self.base_linear_velocity, self.base_angular_velocity
            )

    @property
    def base_angular_velocity(self):
        return self._base_angular_velocity

    @base_angular_velocity.setter
    def base_angular_velocity(self, values):
        self._base_angular_velocity = values
        if hasattr(self, "client_id"):
            self.client_id.resetBaseVelocity(
                self.id, self.base_linear_velocity, self.base_angular_velocity
            )

    def reset(self):
        super().reset()
        self.base_linear_velocity = self.init_base_linear_velocity
        self.base_angular_velocity = self.init_base_angular_velocity

    @abstractmethod
    def load2client(self, client_id):
        raise NotImplementedError(
            "The method load2client() is not implemented in the abstract class."
        )
