import torch
from torch_kinematics_tree.geometrics.frame import Frame


def tensor_circle(pt, radius, tensor=None, device='cpu'):
    if(tensor is None):
        tensor = torch.empty(3, device=device)
    tensor[:2] = torch.as_tensor(pt, device=device)
    tensor[2] = radius
    return tensor


def tensor_sphere(pt, radius, tensor=None, device='cpu'):
    if(tensor is None):
        tensor = torch.empty(4, device=device)
    tensor[:3] = torch.as_tensor(pt, device=device)
    tensor[3] = radius
    return tensor


def tensor_capsule(base, tip, radius, tensor=None, device='cpu'):
    if(tensor is None):
        tensor = torch.empty(7, device=device)
    tensor[:3] = torch.as_tensor(base, device=device)
    tensor[3:6] = torch.as_tensor(tip, device=device)
    tensor[6] = radius
    return tensor


def tensor_cube(pose, dims, device='cpu'):
    w_T_b = Frame(pose=pose, device=device)
    b_T_w = w_T_b.inverse()
    dims_t = torch.tensor([dims[0], dims[1], dims[2]], device=device)
    cube = {'trans': w_T_b.translation(), 'rot': w_T_b.rotation(),
            'inv_trans': b_T_w.translation(), 'inv_rot': b_T_w.rotation(),
            'dims':dims_t}
    cube = [w_T_b.translation(), w_T_b.rotation(),
            b_T_w.translation(), b_T_w.rotation(),
            dims_t]
    return cube