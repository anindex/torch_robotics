import time
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda, DifferentiableTiagoDualHolo
from torch_planning_objectives.fields.collision_bodies import TiagoSphereDistanceField, PandaSphereDistanceField
import torch


if __name__ == "__main__":
    device = 'cpu'
    batch_size = 20
    obstacle_spheres = torch.zeros((1, 4, 4), device=device)
    obstacle_spheres[:, :, -1] = 1.
    obstacle_spheres[:, :, 0] = torch.arange(0, 4, device=device)

    print("===========================Panda Model===============================")
    ## Panda Kinematics ##
    panda = DifferentiableFrankaPanda(device=device)
    q = torch.randn(batch_size,  panda._n_dofs).float().to(device).requires_grad_(True)

    ## Get position-rotation links ##
    links_dict = panda.compute_forward_kinematics_all_links(q, return_dict=True)

    panda_collision = PandaSphereDistanceField(device=device)
    time_start = time.time()
    panda_collision.build_batch_features(batch_dim=[batch_size,], clone_objs=True)
    error = panda_collision.compute_distance(links_dict, obstacle_spheres=obstacle_spheres)
    J = -1. * torch.autograd.grad(error.sum(), q)[0]
    panda_collision.zero_grad()
    print(error)
    time_end = time.time()
    print(f"Computational Time: {time_end - time_start}")

    print("===========================Tiago Model===============================")
    ## Tiago Kinematics ##
    tiago = DifferentiableTiagoDualHolo(device=device)
    q = torch.randn(batch_size,  tiago._n_dofs).float().to(device).requires_grad_(True)

    ## Get position-rotation links ##
    links_dict = tiago.compute_forward_kinematics_all_links(q, return_dict=True)

    tiago_collision = TiagoSphereDistanceField(device=device)
    time_start = time.time()
    tiago_collision.build_batch_features(batch_dim=[batch_size,], clone_objs=True)
    error = tiago_collision.compute_distance(links_dict, obstacle_spheres=obstacle_spheres)
    J = -1. * torch.autograd.grad(error.sum(), q)[0]
    tiago_collision.zero_grad()
    print(error)
    time_end = time.time()
    print(f"Computational Time: {time_end - time_start}")
