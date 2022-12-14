import os
from typing import Union

import numpy as np

from robot_envs.pybullet.objects.core_object import BodyCore
from torch_kinematics_tree.utils.files import get_urdf_path

BOX_ROLES = {-1: "OBSTACLE_BOX", 0: "START_BOX", 1: "GOAL_BOX"}
BOX_COLOR = {
    "START_BOX": [1.0, 0.5, 0.0, 1.0],
    "GOAL_BOX": [0.0, 1.0, 0.0, 1.0],
    "OBSTACLE_BOX": [1.0, 1.0, 1.0, 1.0],
}


class Box(BodyCore):
    def __init__(
        self, base_position: Union[np.ndarray, list], scale: float = 1.0
    ) -> None:
        super(Box, self).__init__(
            base_position=base_position,
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            scale=scale,
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
                    rgbaColor=BOX_COLOR[BOX_ROLES[self.role]],
                )
                for i in range(-1, 4)
            ]

    def load2client(self, client_id):
        path = (get_urdf_path() / 'objects' / 'box_simple.urdf').as_posix()
        self.id = client_id.loadURDF(
            path,
            basePosition=self._base_position,
            baseOrientation=self._base_orientation,
            useFixedBase=self.fixed_base,
            globalScaling=self.scale,
        )
        setattr(self, "client_id", client_id)
        return self.id
