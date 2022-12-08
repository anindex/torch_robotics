import os
from typing import Union

import numpy as np

from robot_envs.pybullet.objects.core_object import BodyCore, DynamicBodyCore
from torch_kinematics_tree.utils.files import get_urdf_path

SPHERE_ROLES = {0: "STATIC_SPHERE", 1: "DYNAMIC_SPHERE"}
SPHERE_COLOR = {
    "STATIC_SPHERE": [1.0, 0.0, 0.0, 1.0],
    "DYNAMIC_SPHERE": [0.5, 0.0, 0.0, 1.0],
}


class Sphere(DynamicBodyCore):
    def __init__(
        self,
        base_position: Union[np.ndarray, list],
        base_linear_velocity: Union[np.ndarray, list],
        scale: float = 0.3,
    ) -> None:
        super(Sphere, self).__init__(
            base_position=base_position,
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=[0.0, 0.0, 0.0],
            scale=scale,
            fixed_base=True,
        )
        self._role = None

    @property
    def role(self) -> Union[None, int]:
        return self._role

    @role.setter
    def role(self, value: int) -> None:
        self._role = value

    def reset(self, role: Union[None, int] = None):
        super().reset()
        self.role = role
        if self.role is not None and hasattr(self, "client_id"):
            [
                self.client_id.changeVisualShape(
                    self.id,
                    i,
                    rgbaColor=SPHERE_COLOR[SPHERE_ROLES[self.role]],
                )
                for i in range(-1, 4)
            ]

    def load2client(self, client_id):
        path = (get_urdf_path() / 'objects' / 'sphere_simple.urdf').as_posix()
        self.id = client_id.loadURDF(
            path,
            basePosition=self._base_position,
            baseOrientation=self._base_orientation,
            useFixedBase=self.fixed_base,
            globalScaling=self.scale,
        )
        setattr(self, "client_id", client_id)
        return self.id
