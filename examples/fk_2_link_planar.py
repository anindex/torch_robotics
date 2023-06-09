import torch
import time

from torch_robotics.torch_kinematics_tree.models.robots import Differentiable2LinkPlanar

if __name__ == "__main__":

    batch_size = 10
    device = "cpu"
    print("===========================2 Link Planar Model===============================")
    planar_kin = Differentiable2LinkPlanar(device=device)
    planar_kin.print_link_names()
    print(planar_kin.get_joint_limits())
    print(planar_kin._n_dofs)
    time_start = time.time()
    q = torch.rand(batch_size, planar_kin._n_dofs).to(device)
    q.requires_grad_(True)
    data = planar_kin.compute_forward_kinematics_all_links(q)
    print(data.shape)
    lin_jacs, ang_jacs = planar_kin.compute_endeffector_jacobian(q, 'link_ee')
    time_end = time.time()
    print(lin_jacs.shape)
    print(ang_jacs.shape)
    print("Computational Time {}".format(time_end - time_start))
