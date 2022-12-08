import numpy as np


def random_init_static_sphere(
    scale_min: float,
    scale_max: float,
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    base_offset: float,
) -> tuple:
    # Get scale
    alpha_scale = np.random.uniform()
    scale = alpha_scale * scale_min + (1 - alpha_scale) * scale_max

    # Get position
    idx = np.random.permutation([1, 0, 0])
    base_position = np.random.rand(3)
    alpha = np.random.rand(1)
    base_position[idx == 1] = (
        alpha * base_position_min[idx == 1] + (1 - alpha) * base_position_max[idx == 1]
    )
    base_position[:-1] *= np.random.randint(2, size=2) * 2 - 1

    # Guarantee no collision at the beginning
    base_position = np.sign(base_position) * np.clip(
        np.abs(base_position), a_min=base_offset, a_max=base_position_max
    )
    # Guarantee no collision with the box
    base_position = np.sign(base_position) * np.clip(
        np.abs(base_position), a_min=base_offset, a_max=base_position_max
    )
    return scale, base_position


def random_init_dynamic_sphere(
    scale_min: float,
    scale_max: float,
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    base_linear_velocity_min: np.ndarray,
    base_linear_velocity_max: np.ndarray,
    base_offset: float,
) -> tuple:
    scale, base_position = random_init_static_sphere(
        scale_min=scale_min,
        scale_max=scale_max,
        base_position_min=base_position_min,
        base_position_max=base_position_max,
        base_offset=base_offset,
    )
    # Get linear velocity
    v_orientation = np.random.randint(2) * 2 - 1
    alpha = np.random.uniform(size=3)
    v_magnitude = (
        alpha * base_linear_velocity_min + (1 - alpha) * base_linear_velocity_max
    )
    base_linear_velocity = v_magnitude * v_orientation
    return scale, base_position, base_linear_velocity


def random_init_static_sphere_simple(
    scale_min: float,
    scale_max: float,
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    shift_order=None,
    offset: float = 0.01,
) -> tuple:
    # Get scale
    alpha_scale = np.random.uniform()
    scale = alpha_scale * scale_min + (1 - alpha_scale) * scale_max

    # Get location
    if shift_order is None:
        location = np.random.randint(0, 4)
        order = np.random.randint(0, 2)
    else:
        location = shift_order[0]
        order = shift_order[1]

    alpha_base_position = np.random.uniform(size=3)
    # north
    if location == 0:
        if order == 0:
            x = alpha_base_position[0] * (base_position_min[1] + scale + offset) + (
                1 - alpha_base_position[0]
            ) * (0 - scale - offset)
        else:
            x = alpha_base_position[0] * (0 + scale + offset) + (
                1 - alpha_base_position[0]
            ) * (base_position_max[1] - scale - offset)
        y = alpha_base_position[1] * (base_position_min[0] + scale + offset) + (
            1 - alpha_base_position[1]
        ) * (base_position_max[0] - scale - offset)
    # east
    elif location == 1:
        x = -(
            alpha_base_position[0] * (base_position_min[0] + scale + offset)
            + (1 - alpha_base_position[0]) * (base_position_max[0] - scale - offset)
        )
        if order == 0:
            y = alpha_base_position[1] * (base_position_min[1] + scale + offset) + (
                1 - alpha_base_position[1]
            ) * (-scale - offset)
        else:
            y = alpha_base_position[1] * (scale + offset) + (
                1 - alpha_base_position[1]
            ) * (base_position_max[1] - scale - offset)
    # south
    elif location == 2:
        if order == 0:
            x = alpha_base_position[0] * (0 + scale + offset) + (
                1 - alpha_base_position[0]
            ) * (base_position_max[1] - scale - offset)
        else:
            x = alpha_base_position[0] * (base_position_min[1] + scale + offset) + (
                1 - alpha_base_position[0]
            ) * (0 - scale - offset)
        y = -(
            alpha_base_position[1] * (base_position_min[0] + scale + offset)
            + (1 - alpha_base_position[1]) * (base_position_max[0])
            - scale
            - offset
        )
    # west
    else:
        x = alpha_base_position[0] * (base_position_min[0] + scale + offset) + (
            1 - alpha_base_position[0]
        ) * (base_position_max[0] - scale - offset)
        if order == 0:
            y = alpha_base_position[1] * (scale + offset) + (
                1 - alpha_base_position[1]
            ) * (base_position_max[1] - scale - offset)
        else:
            y = alpha_base_position[1] * (base_position_min[1] + scale + offset) + (
                1 - alpha_base_position[1]
            ) * (-scale - offset)
    z = alpha_base_position[2] * (base_position_min[2] + scale) + (
        1 - alpha_base_position[2]
    ) * (base_position_max[2] - scale)
    base_position = np.array([x, y, z])
    return scale, base_position


def random_init_dynamic_sphere_simple(
    scale_min: float,
    scale_max: float,
    base_position_min: np.ndarray,
    base_position_max: np.ndarray,
    base_linear_velocity_min: np.ndarray,
    base_linear_velocity_max: np.ndarray,
    shift_order=None,
) -> tuple:

    if shift_order is None:
        location = np.random.randint(0, 4)
        order = np.random.randint(0, 2)
        shift_order = [location, order]
    else:
        location = shift_order[0]

    scale, base_position = random_init_static_sphere_simple(
        scale_min=scale_min,
        scale_max=scale_max,
        base_position_min=base_position_min,
        base_position_max=base_position_max,
        shift_order=shift_order,
    )

    alpha_base_velocity = np.random.uniform(size=3)
    # north and south
    if location in [0, 2]:
        v_x = (
            alpha_base_velocity[0] * base_linear_velocity_min[0]
            + (1 - alpha_base_velocity[0]) * base_linear_velocity_max[0]
        )
        v_y = (
            alpha_base_velocity[1] * base_linear_velocity_min[1]
            + (1 - alpha_base_velocity[1]) * base_linear_velocity_max[1]
        )
    # east and west
    else:
        v_x = (
            alpha_base_velocity[0] * base_linear_velocity_min[1]
            + (1 - alpha_base_velocity[0]) * base_linear_velocity_max[1]
        )
        v_y = (
            alpha_base_velocity[1] * base_linear_velocity_min[0]
            + (1 - alpha_base_velocity[1]) * base_linear_velocity_max[0]
        )
    v_z = (
        alpha_base_velocity[2] * base_linear_velocity_min[2]
        + (1 - alpha_base_velocity[2]) * base_linear_velocity_max[2]
    )
    v_magnitude = np.array([v_x, v_y, v_z])

    # Get linear velocity
    v_orientation = np.random.randint(2, size=3) * 2 - 1
    base_linear_velocity = v_magnitude * v_orientation
    return scale, base_position, base_linear_velocity
