from typing import Union

import numpy as np


def update_linear_velocity_sphere(
    base_position: Union[np.ndarray, list],
    base_linear_velocity: Union[np.ndarray, list],
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    z_offset: float,
) -> tuple:
    if not isinstance(base_position, np.ndarray):
        base_position = np.array(base_position)
    if not isinstance(base_linear_velocity, np.ndarray):
        base_linear_velocity = np.array(base_linear_velocity)

    base_position_new = base_position.copy()
    base_linear_velocity_new = base_linear_velocity.copy()

    if np.max(np.abs(base_position) / base_position_min) <= 1 or 1 <= np.max(
        np.abs(base_position) / base_position_max
    ):
        if np.max(np.abs(base_position) / base_position_min) <= 1:
            idx = np.argmin(1 - np.abs(base_position) / base_position_min)
            base_position_new[idx] = (
                np.sign(base_position_new[idx]) * base_position_min[idx]
            )
            base_linear_velocity_new[idx] = -base_linear_velocity_new[idx]
        else:
            idx = np.argmax(np.abs(base_position) / base_position_max - 1)
            base_position_new[idx] = (
                np.sign(base_position_new[idx]) * base_position_max[idx]
            )
            base_linear_velocity_new[idx] = -base_linear_velocity_new[idx]

    if base_position_new[-1] <= z_offset:
        base_position_new[-1] = z_offset
        base_linear_velocity_new[-1] = np.abs(base_linear_velocity_new[-1])
    return base_position_new, base_linear_velocity_new


def update_linear_velocity_sphere_simple(
    scale: float,
    base_position: Union[np.ndarray, list],
    base_linear_velocity: Union[np.ndarray, list],
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    shift_order: list,
    loc: str = None,
) -> tuple:
    if not isinstance(base_position, np.ndarray):
        base_position = np.array(base_position)
    if not isinstance(base_linear_velocity, np.ndarray):
        base_linear_velocity = np.array(base_linear_velocity)

    base_position_new = base_position.copy()
    base_linear_velocity_new = base_linear_velocity.copy()

    location, order = shift_order
    # north
    if location == 0:
        if order == 0:
            if (
                base_position[0] < (base_position_min[1] + scale)
                or base_position[0] > -scale
            ):
                base_linear_velocity_new[0] = -base_linear_velocity[0]
            base_position_new[0] = np.clip(
                base_position[0], a_min=base_position_min[1] + scale, a_max=-scale
            )
        else:
            if base_position[0] < scale or base_position[0] > (
                base_position_max[1] - scale
            ):
                base_linear_velocity_new[0] = -base_linear_velocity[0]
            base_position_new[0] = np.clip(
                base_position[0], a_min=scale, a_max=base_position_max[1] - scale
            )
        if base_position[1] < (base_position_min[0] + scale) or base_position[1] > (
            base_position_max[0] - scale
        ):
            base_linear_velocity_new[1] = -base_linear_velocity[1]
        base_position_new[1] = np.clip(
            base_position[1],
            a_min=base_position_min[0] + scale,
            a_max=base_position_max[0] - scale,
        )
    # east
    elif location == 1:
        if base_position[0] < -(base_position_max[0] - scale) or base_position[1] > -(
            base_position_min[0] + scale
        ):
            base_linear_velocity_new[0] = -base_linear_velocity[0]
        base_position_new[0] = np.clip(
            base_position[0],
            a_min=-(base_position_max[0] - scale),
            a_max=-(base_position_min[0] + scale),
        )
        if order == 0:
            if (
                base_position[1] < (base_position_min[1] + scale)
                or base_position[1] > -scale
            ):
                base_linear_velocity_new[1] = -base_linear_velocity[1]
            base_position_new[1] = np.clip(
                base_position[1], a_min=base_position_min[1] + scale, a_max=-scale
            )
        else:
            if base_position[1] < scale or base_position[1] > (
                base_position_min[1] - scale
            ):
                base_linear_velocity_new[1] = -base_linear_velocity[1]
            base_position_new[1] = np.clip(
                base_position[1], a_min=scale, a_max=base_position_max[1] - scale
            )
    # south
    elif location == 2:
        if order == 0:
            if base_position[0] < scale or base_position[0] > (
                base_position_max[1] - scale
            ):
                base_linear_velocity_new[0] = -base_linear_velocity[0]
            base_position_new[0] = np.clip(
                base_position[0], a_min=scale, a_max=base_position_max[1] - scale
            )
        else:
            if (
                base_position[0] < (base_position_min[1] + scale)
                or base_position[0] > -scale
            ):
                base_linear_velocity_new[0] = -base_linear_velocity[0]
            base_position_new[0] = np.clip(
                base_position[0], a_min=base_position_min[1] + scale, a_max=-scale
            )
        if base_position[1] < -(base_position_max[0] - scale) or base_position[0] > -(
            base_position_min[0] + scale
        ):
            base_linear_velocity_new[1] = -base_linear_velocity[1]
        base_position_new[1] = np.clip(
            base_position[1],
            a_min=-(base_position_max[0] - scale),
            a_max=-(base_position_min[0] + scale),
        )
    # west
    else:
        if base_position[0] < (base_position_min[0] + scale) or base_position[0] > (
            base_position_max[0] - scale
        ):
            base_linear_velocity_new[0] = -base_linear_velocity[0]
        base_position_new[0] = np.clip(
            base_position[0],
            a_min=base_position_min[0] + scale,
            a_max=base_position_max[0] - scale,
        )
        if order == 0:
            if base_position[1] < scale or base_position[1] > (
                base_position_max[1] - scale
            ):
                base_linear_velocity_new[1] = -base_linear_velocity[1]
            base_position_new[1] = np.clip(
                base_position[1], a_min=scale, a_max=base_position_max[1] - scale
            )
        else:
            if (
                base_position[1] < (base_position_min[1] + scale)
                or base_position[1] > -scale
            ):
                base_linear_velocity_new[1] = -base_linear_velocity[1]
            base_position_new[1] = np.clip(
                base_position[1], a_min=base_position_min[1] + scale, a_max=-scale
            )
    if base_position[2] < (base_position_min[2] + scale) or base_position[2] > (
        base_position_max[2] - scale
    ):
        base_linear_velocity_new[2] = -base_linear_velocity[2]
    base_position_new[2] = np.clip(
        base_position[2],
        a_min=base_position_min[2] + scale,
        a_max=base_position_max[2] - scale,
    )

    #
    # if np.max(np.abs(base_position) / base_position_min) <= 1 or 1 <= np.max(
    #     np.abs(base_position) / base_position_max
    # ):
    #     if np.max(np.abs(base_position) / base_position_min) <= 1:
    #         idx = np.argmin(1 - np.abs(base_position) / base_position_min)
    #         base_position_new[idx] = (
    #             np.sign(base_position_new[idx]) * base_position_min[idx]
    #         )
    #         base_linear_velocity_new[idx] = -base_linear_velocity_new[idx]
    #     else:
    #         idx = np.argmax(np.abs(base_position) / base_position_max - 1)
    #         base_position_new[idx] = (
    #             np.sign(base_position_new[idx]) * base_position_max[idx]
    #         )
    #         base_linear_velocity_new[idx] = -base_linear_velocity_new[idx]
    #
    # if base_position_new[-1] <= z_offset:
    #     base_position_new[-1] = z_offset
    #     base_linear_velocity_new[-1] = np.abs(base_linear_velocity_new[-1])
    return base_position_new, base_linear_velocity_new
