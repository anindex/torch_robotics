import torch
from typing import List, Optional


@torch.jit.script
def compute_spheres_distance(spheres_1, spheres_2):

    b, n, _ = spheres_1.shape
    b_l, n_l, _ = spheres_2.shape

    j = 0
    link_sphere_pts = spheres_1[:,j,:]
    link_sphere_pts = link_sphere_pts.unsqueeze(1)

    # find closest distance to other link spheres:
    s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
    s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
    max_dist = torch.max(s_dist, dim=-1)[0]

    for j in range(1,n):
        link_sphere_pts = spheres_1[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
        s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
        s_dist = torch.max(s_dist, dim=-1)[0]
        max_dist = torch.maximum(max_dist, s_dist)
    dist = max_dist
    return dist


@torch.jit.script
def find_closest_distance(link_idx: int, links_sphere_list: List[torch.Tensor]) -> torch.Tensor:
    """closet distance computed via iteration between sphere sets.
    """

    spheres = links_sphere_list[link_idx]
    b, n, _ = spheres.shape

    dist = torch.zeros((b,len(links_sphere_list), n), device=spheres.device,
                       dtype=spheres.dtype)
    for j in range(n):
        # for every sphere in current link
        link_sphere_pts = spheres[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:

        for i in range(len(links_sphere_list)):
            if(i == link_idx or i==link_idx-1 or i==link_idx+1):
                dist[:,i,j] = -100.0
                continue
            # transform link_idx spheres to current link frame:
            # given a link and another link, find closest distance between them:
            l_spheres = links_sphere_list[i]
            s_dist = torch.norm(l_spheres[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
            s_dist = l_spheres[:,:,3] + link_sphere_pts[:,:,3] - s_dist 

            # dist: b, n_l -> b
            dist[:, i, j] = torch.max(s_dist, dim=-1)[0]
    link_dist = torch.max(dist,dim=-1)[0]
    return link_dist


@torch.jit.script
def find_link_distance(links_sphere_list: List[torch.Tensor], dist: torch.Tensor) -> torch.Tensor:
    futures : List[torch.jit.Future[torch.Tensor]] = []
    n_links = len(links_sphere_list)
    dist *= 0.0
    dist -= 100.0

    for i in range(n_links):
        # for every link, compute the distance to the other links:
        current_spheres = links_sphere_list[i]
        for j in range(i + 2, n_links):
            compute_spheres = links_sphere_list[j]

            # find the distance between the two links:
            d = torch.jit.fork(compute_spheres_distance, current_spheres, compute_spheres)
            futures.append(d)
    k = 0
    for i in range(n_links):
        # for every link, compute the distance to the other links:
        for j in range(i + 2, n_links):
            d = torch.jit.wait(futures[k])
            dist[:, i, j] = d
            dist[:, j, i] = d
            k += 1
    link_dist = torch.max(dist, dim=-1)[0]
    return link_dist


@torch.jit.script
def find_obstacle_distance(obstacle_spheres: torch.Tensor, links_sphere_list: List[torch.Tensor], dist: torch.Tensor) -> torch.Tensor:
    futures : List[torch.jit.Future[torch.Tensor]] = []
    n_links = len(links_sphere_list)
    dist *= 0.0
    dist -= 100.0

    for i in range(n_links):
        # for every link, compute the distance to obstacles
        d = torch.jit.fork(compute_spheres_distance, links_sphere_list[i], obstacle_spheres)
        futures.append(d)

    for i in range(n_links):
        # for every link, compute the distance to obstacles
        dist[:, i] = torch.jit.wait(futures[i])

    link_dist = torch.max(dist, dim=-1)[0]
    return link_dist
