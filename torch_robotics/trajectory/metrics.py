import torch


def compute_path_length(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_smoothness(trajs, robot, trajs_vel=None, dt=1.):
    if trajs_vel is None:
        assert trajs.ndim == 3
        trajs_vel = robot.get_velocity(trajs, dt=dt)
    else:
        assert trajs_vel.ndim == 3
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness


