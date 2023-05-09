import collections

import numpy as np
import torch


DEFAULT_TORCH_ARGS = {'device': 'cpu', 'dtype': torch.float32}

def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def to_numpy(x, dtype=np.float32):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
        return x
    return np.array(x).astype(dtype)


def to_torch(x, device='cpu', dtype=torch.float, requires_grad=False):
    if torch.is_tensor(x):
        return x.clone().to(device=device, dtype=dtype).requires_grad_(requires_grad)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def get_torch_device(device='cuda'):
    if 'cuda' in device and torch.cuda.is_available():
        device = 'cuda'
    elif 'mps' in device:
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)


def freeze_torch_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # If the model is frozen we do not save it again, since the parameters did not change
    model.is_frozen = True


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1)
    if N > 1:
        bcov = bcov / (N - 1)  # Unbiased estimate
    else:
        bcov = bcov / N
    return bcov  # (B, D, D)


def batch_trace(covs):
    return covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


@torch.jit.script
def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int = 10):
    # https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

