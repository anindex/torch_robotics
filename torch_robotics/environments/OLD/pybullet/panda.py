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

from robot_envs.pybullet.objects import Box, Panda, Sphere
from robot_envs.pybullet.utils import (
    random_init_dynamic_sphere_simple,
    random_init_static_sphere_simple,
    update_linear_velocity_sphere_simple,
)

BOX_SCALE = 0.3
BOX_CENTER = 0.5
BOX_POSITION = {
    "NE": [BOX_CENTER, BOX_CENTER, 0.0],
    "NW": [-BOX_CENTER, BOX_CENTER, 0.0],
    "SE": [BOX_CENTER, -BOX_CENTER, 0.0],
    "SW": [-BOX_CENTER, -BOX_CENTER, 0.0],
}

SPHERE_OFFSET = np.array(
    [BOX_CENTER - BOX_SCALE, BOX_CENTER - BOX_SCALE, BOX_CENTER * BOX_SCALE + 0.1]
)
SPHERE_SPACE = {"MIN": np.array([0.4, 0.4, 0.7]), "MAX": np.array([0.6, 0.6, 0.9])}
SPHERE_SCALE = {
    "MIN": 0.08,
    "MAX": 0.1,
}
SPHERE_VELOCITY = {
    "MIN": 0.0,
    "MAX": 0.1,
}


class PandaEnv(object):
    def __init__(
        self, render: bool = False, goal_offset: float = 0.08, **kwargs
    ) -> None:
        self._seed = kwargs.get("seed", None)

        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.render = render
        self._physics_server_initialized = False

        self.t_step = 0
        self._t_start = time.time()
        self._t_H = kwargs.get("horizon", 10000)
        self._frequency = kwargs.get("frequency", 10)
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

        self.num_obst = kwargs.get("num_obst", 2)
        self.max_obs_dist = kwargs.get("max_obs_dist", 0.0)
        self.max_floor_dist = kwargs.get("max_floor_dist", 0.0)
        # Define the motion of the spheres:
        #   (0) Static;
        #   (1) Dynamic
        #   (2) partly static partly dynamics
        self.motion_obstacles = kwargs.get("motion_obstacles", 0)

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

    def reset(self, seed: int = None):
        # Init Physics server
        if not self._physics_server_initialized:
            self._init_physics_server()

        # Set seed
        seed = self._seed if seed is None else seed
        self.seed(seed=seed)

        # Reset Panda
        self.panda.reset()

        # Reset the role of Boxes and start / goal position
        z_offset = np.array([0.0, 0.0, 0.08])
        setattr(self, "shift", np.random.randint(0, 4))
        setattr(self, "order", np.random.randint(0, 2))
        if self.order == 0:
            array = [0, 1, -1, -1]
        else:
            array = [1, 0, -1, -1]
        box_roles = np.roll(array, self.shift)
        for box, role in zip(self.boxes, box_roles):
            box.reset(role=role)
            if role == 0:
                s_T1 = box.base_position + z_offset
            elif role == 1:
                s_T2 = box.base_position + z_offset
        self._s_T = [s_T1, s_T2]

        # Reset the roles, positions and possible velocities of #num_obs Spheres.
        if self.motion_obstacles == 0:
            sphere_roles = np.zeros(self.num_obst)
        elif self.motion_obstacles == 1:
            sphere_roles = np.ones(self.num_obst)
        else:
            sphere_roles = np.random.randint(0, 2, size=self.num_obst)

        for sphere, role in zip(self.spheres, sphere_roles):
            if role == 0:
                _, base_position = random_init_static_sphere_simple(
                    scale_min=SPHERE_SCALE["MIN"],
                    scale_max=SPHERE_SCALE["MAX"],
                    base_position_min=np.array(
                        [
                            BOX_CENTER - 0.6 * BOX_SCALE,
                            -np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.05,
                        ]
                    ),
                    base_position_max=np.array(
                        [
                            BOX_CENTER + 0.6 * BOX_SCALE,
                            np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.5,
                        ]
                    ),
                    shift_order=[self.shift, self.order],
                )
                sphere.init_base_position = base_position
                sphere.init_base_linear_velocity = np.array([0.0, 0.0, 0.0])
            else:
                (
                    _,
                    base_position,
                    base_linear_velocity,
                ) = random_init_dynamic_sphere_simple(
                    scale_min=SPHERE_SCALE["MIN"],
                    scale_max=SPHERE_SCALE["MAX"],
                    base_position_min=np.array(
                        [
                            BOX_CENTER - 0.6 * BOX_SCALE,
                            -np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.05,
                        ]
                    ),
                    base_position_max=np.array(
                        [
                            BOX_CENTER + 0.6 * BOX_SCALE,
                            np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.5,
                        ]
                    ),
                    base_linear_velocity_min=np.array(
                        [
                            SPHERE_VELOCITY["MIN"],
                            SPHERE_VELOCITY["MIN"],
                            SPHERE_VELOCITY["MIN"],
                        ]
                    ),
                    base_linear_velocity_max=np.array(
                        [
                            SPHERE_VELOCITY["MAX"] / 4,
                            SPHERE_VELOCITY["MAX"] / 2,
                            SPHERE_VELOCITY["MAX"],
                        ]
                    ),
                    shift_order=[self.shift, self.order],
                )
                sphere.init_base_position = base_position
                sphere.init_base_linear_velocity = base_linear_velocity
            sphere.reset(role=role)

        # Get obstacle_information
        obs_state = self._state_obstacles()

        # Reset env variables
        self._goal_idx = 0
        self.goal_reached = [False, False]
        self.is_contact = False
        self._done = False
        self.t_step = 0
        self._t_start = time.time()

        self.s_t = [np.array(self.panda.getJointStates()).reshape(1, 1, -1), obs_state]

        self._init_buffer()
        return self.s_t

    def step(self, a_t=None):
        self.t_step += 1

        # Update Panda
        if a_t is None:
            a_t = np.array(self.panda.q)
        self.panda.setTargetPositions(a_t.squeeze())
        # Update Obstacle
        for sphere in self.spheres:
            if sphere.role == 1:
                base_position = self.client_id.getBasePositionAndOrientation(sphere.id)[
                    0
                ]
                base_linear_velocity = self.client_id.getBaseVelocity(sphere.id)[0]

                (
                    base_position_new,
                    base_linear_velocity_new,
                ) = update_linear_velocity_sphere_simple(
                    scale=sphere.scale,
                    base_position=base_position,
                    base_linear_velocity=base_linear_velocity,
                    base_position_min=np.array(
                        [
                            BOX_CENTER - 0.6 * BOX_SCALE,
                            -np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.05,
                        ]
                    ),
                    base_position_max=np.array(
                        [
                            BOX_CENTER + 0.6 * BOX_SCALE,
                            np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.5,
                        ]
                    ),
                    shift_order=[self.shift, self.order],
                )

                sphere.base_position = base_position_new
                sphere.base_linear_velocity = base_linear_velocity_new

        [self.client_id.stepSimulation() for _ in range(self._frequency)]

        self.s_t = [
            np.array(self.panda.getJointStates()).reshape(1, 1, -1).copy(),
            self._state_obstacles().copy(),
        ]
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
                warnings.warn("Connect robots")
                # time.sleep(10)
                self.is_contact = True
        else:
            warnings.warn("Connect floor")
            # time.sleep(10)
            self.is_contact = True

        # Check whether final state is reached or not
        dist2goal = np.sqrt(
            np.sum(
                (self.panda.getEEPositionAndOrientation()[0] - self.s_T.squeeze()) ** 2
            )
        )
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
        self._init_box()
        self._init_spheres()
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
        self.client_id.loadURDF("plane.urdf")

    def _init_box(self):
        boxes = [
            Box(base_position=BOX_POSITION[keys], scale=BOX_SCALE)
            for keys in ["NE", "NW", "SW", "SE"]
        ]
        [box.load2client(self.client_id) for box in boxes]
        self._obstacles.update({"boxes": boxes})

    def _init_spheres(self):
        if self.motion_obstacles == 0:
            roles = np.zeros(self.num_obst)
        elif self.motion_obstacles == 1:
            roles = np.ones(self.num_obst)
        else:
            roles = np.random.randint(0, 2, size=self.num_obst)

        spheres = []
        for role in roles:
            if role == 0:
                scale, base_position = random_init_static_sphere_simple(
                    scale_min=SPHERE_SCALE["MIN"],
                    scale_max=SPHERE_SCALE["MAX"],
                    base_position_min=np.array(
                        [
                            BOX_CENTER - 0.6 * BOX_SCALE,
                            -np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.05,
                        ]
                    ),
                    base_position_max=np.array(
                        [
                            BOX_CENTER + 0.6 * BOX_SCALE,
                            np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.5,
                        ]
                    ),
                )
                spheres.append(
                    Sphere(
                        base_position=base_position,
                        base_linear_velocity=np.array([0.0, 0.0, 0.0]),
                        scale=scale,
                    )
                )
            else:
                (
                    scale,
                    base_position,
                    base_linear_velocity,
                ) = random_init_dynamic_sphere_simple(
                    scale_min=SPHERE_SCALE["MIN"],
                    scale_max=SPHERE_SCALE["MAX"],
                    base_position_min=np.array(
                        [
                            BOX_CENTER - 0.6 * BOX_SCALE,
                            -np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.05,
                        ]
                    ),
                    base_position_max=np.array(
                        [
                            BOX_CENTER + 0.6 * BOX_SCALE,
                            np.abs(BOX_CENTER - 0.5 * BOX_SCALE),
                            0.5,
                        ]
                    ),
                    base_linear_velocity_min=np.array(
                        [
                            SPHERE_VELOCITY["MIN"],
                            SPHERE_VELOCITY["MIN"],
                            SPHERE_VELOCITY["MIN"],
                        ]
                    ),
                    base_linear_velocity_max=np.array(
                        [
                            SPHERE_VELOCITY["MAX"] / 4,
                            SPHERE_VELOCITY["MAX"],
                            SPHERE_VELOCITY["MAX"],
                        ]
                    ),
                )
                spheres.append(
                    Sphere(
                        base_position=base_position,
                        base_linear_velocity=base_linear_velocity,
                        scale=scale,
                    )
                )
        [sphere.load2client(self.client_id) for sphere in spheres]
        self._obstacles.update({"spheres": spheres})

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
                    "s_robot": self.s_t[0].copy(),
                    "a_robot": self.a_t.copy(),
                    "s_obs": self.s_t[1].copy(),
                    "s_goal": self.s_T.copy(),
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
                    "s_robot": self.s_t[0].copy(),
                    "a_robot": self.a_t.copy(),
                    "s_obs": self.s_t[1].copy(),
                    "s_goal": self.s_T.copy(),
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
                    "s_robot": self.s_t[0],
                    "a_robot": self.a_t,
                    "s_obs": self.s_t[1],
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
        delta_goal = ee_position - self.s_T.squeeze()
        dist2goal = np.sqrt(np.sum((delta_goal) ** 2))

        costs = -gain / (dist2goal + eps)
        return np.where(self.is_contact, np.ones_like(costs) * 1e2, costs)

    def _state_obstacles(self) -> np.ndarray:
        boxes_state = np.concatenate(
            (
                np.array(
                    [
                        self.client_id.getBasePositionAndOrientation(box.id)[0]
                        for box in self.boxes
                    ]
                ),
                np.array(
                    [self.client_id.getBaseVelocity(box.id)[0] for box in self.boxes]
                ),
                np.array([box.scale for box in self.boxes])[:, None],
            ),
            axis=-1,
        )[None, :]
        if not any(self.spheres):
            return boxes_state
        else:
            spheres_state = np.concatenate(
                (
                    np.array(
                        [
                            self.client_id.getBasePositionAndOrientation(sphere.id)[0]
                            for sphere in self.spheres
                        ]
                    ),
                    np.array(
                        [
                            self.client_id.getBaseVelocity(sphere.id)[0]
                            for sphere in self.spheres
                        ]
                    ),
                    np.array([sphere.scale for sphere in self.spheres])[:, None],
                ),
                axis=-1,
            )[None, :]
            obs_state = np.concatenate((boxes_state, spheres_state), axis=1)
            return obs_state


if __name__ == '__main__':
    env = PandaEnv(render=True)
    env.reset()
    while True:
        env.step()
