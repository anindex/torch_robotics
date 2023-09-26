import numpy as np
import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


def compute_path_length(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_variance_waypoints(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    parwise_distance_between_points_waypoints = torch.cdist(trajs_pos, trajs_pos, p=2)

    sum_var_waypoints = 0.
    for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
        parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
        distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
        sum_var_waypoints += torch.var(distances)
    return sum_var_waypoints


def compute_smoothness(trajs, robot, trajs_vel=None):
    if trajs_vel is None:
        assert trajs.ndim == 3
        trajs_vel = robot.get_velocity(trajs)
    else:
        assert trajs_vel.ndim == 3
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness
