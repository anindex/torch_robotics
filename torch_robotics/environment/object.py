import torch


class Object:

    def __init__(self, object_field, name='obj', pos=None, ori=None):
        self.object_field = object_field

        # position and orientation (wxyz quaternion)
        assert (pos is None and ori is None) or (pos.ndims == 2 and ori.ndims == 2)
        self.pos = torch.zeros((1, self.object_field.dim), **self.object_field.tensor_args) if pos is None else pos
        self.ori = torch.tensor([1, 0, 0, 0], **self.object_field.tensor_args).view((1, -1)) if ori is None else ori

