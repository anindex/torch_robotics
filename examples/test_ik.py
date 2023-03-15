import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_kinematics_tree.geometrics.frame import Frame
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_kinematics_tree.geometrics.spatial_vector import (
    z_rot,
    y_rot,
    x_rot,
)


if __name__ == "__main__":
    batch_size = 10
    # device = 'cuda'
    device = 'cpu'
    tensor_args = dict(device=device, dtype=torch.float32)

    target_pos = np.array([0.3, 0.3, 0.3])
    target_rot = (z_rot(-torch.tensor(torch.pi)) @ y_rot(-torch.tensor(torch.pi))).to(**tensor_args)
    target_frame = Frame(rot=target_rot, trans=torch.from_numpy(target_pos).to(**tensor_args), device=device)
    target_H = target_frame.get_transform_matrix()  # set translation and orientation of target here

    print("===========================Panda Model===============================")
    ## Panda Kinematics ##
    panda = DifferentiableFrankaPanda(gripper=False, device=device)
    
    time_start = time.time()
    q_ik = panda.inverse_kinematics(target_H, link_name='ee_link', batch_size=batch_size, max_iters=2000, lr=1e-1, eps=1e-4)
    time_end = time.time()
    print(f"Computational Time: {time_end - time_start}")

    # plot the result
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal')
    for i in range(batch_size):
        skeleton = get_skeleton_from_model(panda, q_ik[i], panda.get_link_names()) # visualize IK solution
        skeleton.draw_skeleton(ax=ax)
    ax.plot(target_pos[0], target_pos[1], target_pos[2], 'r*', markersize=20)
    plt.show()
