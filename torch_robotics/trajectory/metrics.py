import torch


def compute_path_length(trajs, env):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = env.get_q_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_smoothness(trajs, env):
    assert trajs.ndim == 3
    if trajs.shape[-1] == env.q_dim:
        # if there is no velocity information in the trajectory, compute it via finite difference
        trajs_pos = env.get_q_position(trajs)
        trajs_vel = torch.diff(trajs_pos, dim=-2)
    else:
        trajs_vel = env.get_q_velocity(trajs)
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1).mean(-1)  # mean over trajectory horizon
    return smoothness


