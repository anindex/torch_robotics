import abc

import torch
from urdf_parser_py.urdf import Box

from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import to_numpy


class GraspedObject(ObjectField):

    def __init__(self, primitive_fields, object_collision_margin=0.001, **kwargs):
        assert len(primitive_fields) == 1
        super().__init__(primitive_fields, **kwargs)

        self.name = self.__class__.__name__

        self.object_collision_margin = object_collision_margin

        # Geometry URDF
        self.geometry_urdf = self.get_geometry_urdf()

    def get_geometry_urdf(self):
        primitive_field = self.fields[0]

        if isinstance(primitive_field, MultiBoxField):
            size = primitive_field.sizes[0]
            return Box(to_numpy(size))
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def get_points_for_collision(self):
        raise NotImplementedError


class GraspedObjectBox(GraspedObject):

    def __init__(self, attached_to_frame, tensor_args=None, **kwargs):
        # Box sizes
        primitive_fields = [
            MultiBoxField(torch.zeros(3, **tensor_args).view(1, -1),
                          torch.tensor([0.05, 0.05, 0.15], **tensor_args).view(1, -1),
                          tensor_args=tensor_args)
        ]

        # Default position and orientation of the object center wrt to some frame of the robot (e.g., panda_hand)
        pos = torch.tensor([0., 0., 0.11], **tensor_args)
        ori = torch.tensor([0, 0.7071081, 0, 0.7071055], **tensor_args)
        super().__init__(
            primitive_fields,
            pos=pos, ori=ori, reference_frame=attached_to_frame,
            **kwargs)

        self.points_for_collision = self.get_points_for_collision()
        self.n_points_for_collision = len(self.points_for_collision)

    def get_points_for_collision(self):
        # Points on the box vertices and centers of faces
        # These points are added to the robot's urdf for collision checking
        size = self.fields[0].sizes[0]
        x, y, z = size
        vertices = torch.tensor(
            [
                [x/2, y/2, -z/2],
                [x/2, -y/2, -z/2],
                [-x/2, -y/2, -z/2],
                [-x/2, y/2, -z/2],
                [x/2, y/2, z/2],
                [x/2, -y/2, z/2],
                [-x/2, -y/2, z/2],
                [-x/2, y/2, z/2],
            ],
            **self.tensor_args)

        faces = torch.tensor(
            [
                [x/2, 0, 0],
                [0, -y/2, 0],
                [-x/2, 0, 0],
                [0, y/2, 0],
                [0, 0, z/2],
                [0, 0, -z/2],
            ],
            **self.tensor_args)

        points = torch.cat((vertices, faces), dim=0)
        return points
