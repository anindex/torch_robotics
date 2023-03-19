"""
This Franka Panda class was adapted from:
https://github.com/bryandlee/franka_pybullet/tree/ac86319a0b2f6c863ba3c7ee3d52f4f51b2be3bd
"""

import os
import time
from copy import copy
from typing import Union

import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import torch

from robot_envs.pybullet.objects import Box, Panda, Sphere
from robot_envs.pybullet.objects.box_object import BoxBullet
from robot_envs.pybullet.objects.sphere_object import SphereBullet
from robot_envs.pybullet.utils import (
    random_init_dynamic_sphere_simple,
    random_init_static_sphere_simple,
    update_linear_velocity_sphere_simple,
)
from torch_planning_objectives.fields.primitive_distance_fields import SphereField, BoxField



class PandaEnvPyBulletBase(object):
    def __init__(
        self, render: bool = False, goal_offset: float = 0.08,
            obst_primitives_l = None,
            obst_primitives_extra_index_l=None,
            **kwargs
    ) -> None:
        self._seed = kwargs.get("seed", None)

        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.render = render
        self._physics_server_initialized = False

        self.t_step = 0
        self._t_start = time.time()
        self._t_H = kwargs.get("horizon", 10000)
        self._frequency = kwargs.get("frequency", 1000)
        self.realtime = kwargs.get("realtime", False)

        self.a_t = None
        self.s_t = None
        self._s_T = [None, None]
        self._goals_reached = []
        self._goal_offset = np.array([0.0, 0.0, goal_offset])
        self._goal_idx = 0
        self._goal_metric = np.eye(3)
        self.goal_reached = [False, False]
        self.is_contact = False
        self._done = False

        # Obstacles
        self.obst_primitives_l = obst_primitives_l
        self.obst_primitives_extra_index_l = obst_primitives_extra_index_l

        self.max_obs_dist = kwargs.get("max_obs_dist", 0.0)
        self.max_floor_dist = kwargs.get("max_floor_dist", 0.0)

        self._buffer_goal_counter = 1
        self._max_buffer_len = int(kwargs.get("buffer_length", 1000))
        self._init_buffer()

    @property
    def buffer(self) -> list:
        return self._buffer[: self._buffer_idx]

    @property
    def obstacles(self) -> dict:
        return self._obstacles

    @obstacles.setter
    def obstacles(self, values: dict) -> None:
        self._obstacles = values

    @property
    def boxes(self):
        return self.obstacles.get("boxes", None)

    @boxes.setter
    def boxes(self, values: list) -> None:
        self.obstacles.update({"boxes": values})

    @property
    def spheres(self):
        return self.obstacles.get("spheres", None)

    @spheres.setter
    def spheres(self, values: list) -> None:
        self.obstacles.update({"spheres": values})

    @property
    def done(self) -> np.ndarray:
        return self._done

    @property
    def s_T(self) -> Union[np.ndarray, None]:
        if self._s_T[self._goal_idx] is not None:
            return self._s_T[self._goal_idx][None, None, :]
        else:
            return self._s_T[self._goal_idx]

    def seed(self, seed: int = None) -> list:
        np.random.seed(seed)
        return [seed]

    def not_t_horizon(self):
        if self.realtime:
            return np.abs(time.time() - self._t_start) < self._t_H
        else:
            return self.t_step < self._t_H

    def reset(self, robot_q=None, target_EE=None, seed: int = None, **kwargs):
        # Init Physics server
        if not self._physics_server_initialized:
            self._init_physics_server()

        # Set seed
        seed = self._seed if seed is None else seed
        self.seed(seed=seed)

        # Reset Panda
        self.panda.reset(robot_q)

        self._init_target_EE(target_EE)

        # Reset env variables
        self._goal_idx = 0
        self.goal_reached = [False, False]
        self.is_contact = False
        self._done = False
        self.t_step = 0
        self._t_start = time.time()

        self.s_t = np.array(self.panda.getJointStates()).reshape((1, 1, -1))

        self._init_buffer()
        return self.s_t

    def step(self, a_t=None):
        self.t_step += 1

        # Update Panda
        if a_t is None:
            a_t = np.array(self.panda.q)
        self.panda.setTargetPositions(a_t.squeeze())

        for _ in range(self._frequency):
            self.client_id.stepSimulation()
        # Wait dt for visualization
        # time.sleep(1 / self._frequency)


        self.s_t = np.array(self.panda.getJointStates()).reshape(1, 1, -1).copy()
        self.a_t = a_t.copy()
        self.is_contact = False

        # Check collision
        import warnings

        if (
            len(self.client_id.getClosestPoints(self.panda.id, 0, self.max_floor_dist))
            <= 1
        ):
            dist2different_links = self.client_id.getClosestPoints(
                self.panda.id, self.panda.id, self.max_obs_dist
            )
            n_links = self.client_id.getNumJoints(self.panda.id) - 1
            if len(dist2different_links) <= 3 * n_links:
                pass
                for obstacles in self.boxes + self.spheres:
                    dist2obs = self.client_id.getClosestPoints(
                        self.panda.id, obstacles.id, self.max_obs_dist
                    )
                    if len(dist2obs) > 0:
                        warnings.warn("Connect obs")
                        # time.sleep(10)
                        self.is_contact = True
                        break
            else:
                warnings.warn("Connect robot")
                # time.sleep(10)
                self.is_contact = True
        else:
            warnings.warn("Connect floor")
            # time.sleep(10)
            self.is_contact = True

        # Check whether final state is reached or not
        # dist2goal = np.sqrt(
        #     np.sum(
        #         (self.panda.getEEPositionAndOrientation()[0] - self.s_T.squeeze()) ** 2
        #     )
        # )
        dist2goal = 10.
        self.goal_reached[self._goal_idx] = dist2goal < 0.125
        if self.goal_reached[0] and self._goal_idx == 0:
            self._goal_idx = 1

        # Update goal flag
        if self.is_contact | all(self.goal_reached):
            self._done = True

        # Calculate costs
        costs = self.cost_function()

        # Update buffer
        self._update_buffer()

        return (
            self.s_t,
            costs,
            self.done,
            [self.s_T, self.goal_reached, self.is_contact],
        )

    def close(self):
        self.client_id.disconnect()

    def _init_physics_server(self) -> None:
        self._init_client()
        self.panda = Panda()
        self.panda.load2client(self.client_id)
        self._obstacles = dict()
        self._init_obstacles()
        self._physics_server_initialized = True

    def _init_client(self):
        if self.render:
            mode = p.GUI
        else:
            mode = p.DIRECT
        self.client_id = bc.BulletClient(
            connection_mode=mode,
            options="--background_color_red=0. --background_color_green=0. --background_color_blue=0.",
        )
        # Remove image previews from UI
        self.client_id.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0
        )
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        self.client_id.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=50.0, cameraPitch=-35, cameraTargetPosition=[0, 0, 0.0])
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.client_id.setGravity(*np.array([0, 0, -9.81]))
        p.setAdditionalSearchPath(pd.getDataPath())
        # self.client_id.loadURDF("plane.urdf")
        self.client_id.loadURDF(
            "plane.urdf",
            [0, 0, -1.25],
            useFixedBase=True,
        )

    def _init_obstacles(self):
        boxes = []
        spheres = []
        for obstacle_primitive in self.obst_primitives_l:
            if isinstance(obstacle_primitive, BoxField):
                for i, (center, half_size) in enumerate(zip(obstacle_primitive.centers, obstacle_primitive.half_sizes)):
                    extra_option = {}
                    if self.obst_primitives_extra_index_l is not None:
                        if i in self.obst_primitives_extra_index_l:
                            extra_option = {'color': [1, 0, 0, 1]}
                    boxes.append(
                        BoxBullet(base_position=center.tolist(), half_sizes=half_size, scale=1.,
                                  **extra_option)
                    )

            if isinstance(obstacle_primitive, SphereField):
                for i, (center, radius) in enumerate(zip(obstacle_primitive.centers, obstacle_primitive.radii)):
                    extra_option = {}
                    if self.obst_primitives_extra_index_l is not None:
                        if i in self.obst_primitives_extra_index_l:
                            extra_option = {'color': [1, 0, 0, 1]}
                    spheres.append(
                        SphereBullet(
                            base_position=center.tolist(),
                            radius=radius,
                            scale=1.0,
                            **extra_option
                        )
                    )

        [box.load2client(self.client_id) for box in boxes]
        self._obstacles.update({"boxes": boxes})
        [sphere.load2client(self.client_id) for sphere in spheres]
        self._obstacles.update({"spheres": spheres})

    def _init_target_EE(self, target=None):
        if target is None:
            return
        self.target = SphereBullet(
            base_position=target[:3, -1],
            radius=0.02,
            scale=1.0,
            color=[0, 1, 0, 1.]
        )
        rot_mat = target[:3, :3]
        vec = np.array([0, 0, 0.1])
        rot_vec = rot_mat @ vec
        self.client_id.addUserDebugLine(
            target[:3, -1],
            target[:3, -1] + rot_vec,
            lineColorRGB=[0, 1, 0],
            lineWidth=3,
            lifeTime=0,
        )
        self.target.load2client(self.client_id)

    def _init_buffer(self) -> None:
        if getattr(self, "buffer_idx", None):
            self._buffer_idx = 0
            self._buffer = [dict() for _ in range(self._max_buffer_len)]
        else:
            buffer_idx = 0
            buffer = [dict() for _ in range(self._max_buffer_len)]
            setattr(self, "_buffer_idx", buffer_idx)
            setattr(self, "_buffer", buffer)
        return None

    def _update_buffer(self) -> None:
        if self.t_step == 1:
            self._buffer[self._buffer_idx].update(
                {
                    "s_robot": self.s_t.copy(),
                    "a_robot": self.a_t.copy(),
                    # "s_obs": self.s_t[1].copy(),
                    # "s_goal": self.s_T.copy(),
                    "is_contact": copy(self.is_contact),
                    "goal_reached": copy(self.goal_reached),
                    "time_horizon": copy(not self.not_t_horizon()),
                    "time": self.t_step - 1,
                }
            )
            self._buffer_idx += 1
        if self.t_step % 50 == 0:
            self._buffer[self._buffer_idx].update(
                {
                    "s_robot": self.s_t.copy(),
                    "a_robot": self.a_t.copy(),
                    # "s_obs": self.s_t[1].copy(),
                    # "s_goal": self.s_T.copy(),
                    "is_contact": copy(self.is_contact),
                    "goal_reached": copy(self.goal_reached),
                    "time_horizon": copy(not self.not_t_horizon()),
                    "time": self.t_step,
                }
            )
            self._buffer_idx += 1
        if (
            self.is_contact
            or (sum(self.goal_reached) == self._buffer_goal_counter)
            or not self.not_t_horizon()
        ):
            self._buffer[self._buffer_idx].update(
                {
                    "s_robot": self.s_t,
                    "a_robot": self.a_t,
                    # "s_obs": self.s_t[1],
                    "s_goal": self.s_T,
                    "is_contact": copy(self.is_contact),
                    "goal_reached": copy(self.goal_reached),
                    "time_horizon": copy(not self.not_t_horizon()),
                    "time": self.t_step,
                }
            )
            self._buffer_idx += 1
            if sum(self.goal_reached) == self._buffer_goal_counter:
                self._buffer_goal_counter += 1
        if self._buffer_idx >= self._max_buffer_len:
            self._buffer_idx = 0
        return None

    def cost_function(self) -> np.ndarray:
        gain = 1e2
        eps = 1e-6
        ee_position = self.panda.getEEPositionAndOrientation()[0]
        # delta_goal = ee_position - self.s_T.squeeze()
        delta_goal = 10.
        dist2goal = np.sqrt(np.sum((delta_goal) ** 2))

        costs = -gain / (dist2goal + eps)
        return np.where(self.is_contact, np.ones_like(costs) * 1e2, costs)


if __name__ == '__main__':
    tensor_args = dict(device='cpu', dtype=torch.float32)

    obst_primitives_l = [
        SphereField([
            [0.6, 0.3, 0.],
            [0.5, 0.3, 0.5],
            [-0.5, 0.25, 0.6],
            [-0.6, -0.2, 0.4],
            [-0.7, 0.1, 0.0],
            [0.5, -0.45, 0.2],
            [0.6, -0.35, 0.6],
            [0.3, 0.0, 1.0],
        ],
            [
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
                0.15,
            ],
            tensor_args=tensor_args
        )
    ]

    env = PandaEnvPyBulletBase(
        obst_primitives_l=obst_primitives_l,
        render=True
    )
    target = np.eye(4)
    target[:3, -1] = np.array([1, 1, 1])
    env.reset(target_EE=target)
    while True:
        env.step()
