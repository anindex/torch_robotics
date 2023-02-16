import time
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_planning_objectives.fields.distance_fields import EmbodimentDistanceField
from torch_planning_objectives.fields.primitive_distance_fields import MultiSpheres
import torch
import numpy as np


if __name__ == "__main__":
    batch_size = 10
    device = 'cpu'
    tensor_args = dict(device=device, dtype=torch.float32)
    df = MultiSpheres(
        centers=[[1., 1., 1.], [0., 0., 2.], [0., 0., 3.]],
        radii=[0.5, 0.5, 0.5],
        tensor_args=tensor_args
    )

    print("===========================Panda Model===============================")
    ## Panda Kinematics ##
    panda = DifferentiableFrankaPanda(gripper=False, device=device)
    q = torch.randn(batch_size, panda._n_dofs).to(**tensor_args)
    qd = torch.randn(batch_size, panda._n_dofs).to(**tensor_args)

    ee_pos, ee_rot, lin_jac, ang_jac = panda.compute_fk_and_jacobian(q, qd, 'ee_link')
    target_pos = np.array([0.8, 0.8, 0.5])
    euler = np.array([torch.pi / 2, 0.0, 0.0])
    q_sol = panda.inverse_kinematics(target_pos, euler, frame='ee_link', step_rate=0.01, max_iter=1000)
    # print(ee_pos.shape, ee_rot.shape, lin_jac.shape, ang_jac.shape)

    ## Get position-rotation links ##
    link_tensor = panda.compute_forward_kinematics_all_links(q)
    link_pos = link_tensor[..., :3, 3]

    field = EmbodimentDistanceField(self_margin=0.005, obst_margin=0.03, field_type='occupancy', num_interpolate=4, link_interpolate_range=[2, 7])
    time_start = time.time()
    self_dist, obst_dist = field.compute_distance(link_pos, [df])
    self_cost, obst_cost = field.compute_cost(link_pos, [df])
    time_end = time.time()
    print(self_dist, obst_dist)
    print(self_cost, obst_cost)
    print(f"Computational Time: {time_end - time_start}")
