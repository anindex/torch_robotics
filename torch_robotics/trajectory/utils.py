import numpy as np
import torch
from scipy import interpolate

from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy


def smoothen_trajectory(traj, traj_len=30, zero_velocity=True, tensor_args=None):
    traj = to_numpy(traj)
    try:
        # bc_type='clamped' for zero velocities at start and finish
        spline_pos = interpolate.make_interp_spline(np.linspace(0, 1, traj.shape[0]), traj, k=3, bc_type='clamped')
        spline_vel = spline_pos.derivative(1)
    except TypeError:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj = np.vstack((traj, traj[-1] + np.random.normal(0, 0.01)))
        return smoothen_trajectory(traj, traj_len=traj_len, tensor_args=tensor_args)

    pos = spline_pos(np.linspace(0, 1, traj_len))
    if zero_velocity:
        vel = np.zeros_like(pos)
    else:
        vel = spline_vel(np.linspace(0, 1, traj_len))
    return to_torch(pos, **tensor_args), to_torch(vel, **tensor_args)


def interpolate_traj_via_points(trajs, num_intepolation=10):
    # Interpolates a trajectory linearly between waypoints
    H, D = trajs.shape[-2:]
    if num_intepolation > 0:
        assert trajs.ndim > 1
        traj_dim = trajs.shape
        alpha = torch.linspace(0, 1, num_intepolation + 2).type_as(trajs)[1:num_intepolation + 1]
        alpha = alpha.view((1,) * len(traj_dim[:-1]) + (-1, 1))
        interpolated_trajs = trajs[..., 0:traj_dim[-2] - 1, None, :] * alpha + \
                             trajs[..., 1:traj_dim[-2], None, :] * (1 - alpha)
        interpolated_trajs = interpolated_trajs.view(traj_dim[:-2] + (-1, D))
    else:
        interpolated_trajs = trajs
    return interpolated_trajs