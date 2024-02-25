import torch
from urdf_parser_py.urdf import URDF
import os

torch.set_default_dtype(torch.float32)


class URDFRobotModel(object):
    def __init__(self, model_path, device='cpu'):
        self.robot = URDF.from_xml_file(model_path)
        self.links = self.robot.links
        self.model_path = model_path
        self.device = device

    def find_joint_of_body(self, body_name):
        for (i, joint) in enumerate(self.robot.joints):
            if joint.child == body_name:
                return i
        return -1

    def find_link_idx(self, link_name):
        for (i, link) in enumerate(self.robot.links):
            if(link.name == link_name):
                return i
        return -1

    def get_name_of_parent_body(self, link_name):
        jid = self.find_joint_of_body(link_name)
        joint = self.robot.joints[jid]
        return joint.parent

    def get_link_collision_mesh(self, link_name):
        idx = self.find_link_idx(link_name)
        link = self.robot.links[idx]
        mesh_fname = link.collision.geometry.filename
        mesh_origin = link.collision.origin
        origin_pose = torch.zeros(6).to(self.device)
        if(mesh_origin is not None):
            origin_pose[:3] = mesh_origin.position
            origin_pose[3:6] = mesh_origin.rotation
            
        # join to urdf path
        mesh_fname = os.path.join(os.path.dirname(self.model_path), mesh_fname)
        return mesh_fname, origin_pose

    def get_body_parameters(self, i, link):
        body_params = {}
        body_params['joint_id'] = i
        body_params['link_name'] = link.name

        if i == 0:
            rot_angles = torch.zeros(3).to(device=self.device)
            trans = torch.zeros(3).to(device=self.device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self.device)
        else:
            link_name = link.name
            jid = self.find_joint_of_body(link_name)
            joint = self.robot.joints[jid]
            joint_name = joint.name
            # find joint that is the "child" of this body according to urdf

            rpy = torch.tensor(joint.origin.rotation, device=self.device)
            rot_angles = torch.tensor([rpy[0], rpy[1], rpy[2]], device=self.device)
            trans = torch.tensor(joint.origin.position, device=self.device)
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, device=self.device)
            joint_axis = torch.zeros((1, 3), device=self.device)
            if joint_type != 'fixed':
                joint_limits = {'effort': joint.limit.effort,
                                'lower': joint.limit.lower,
                                'upper': joint.limit.upper,
                                'velocity': joint.limit.velocity}
                try:
                    joint_damping = torch.tensor(joint.dynamics.damping, device=self.device)
                except AttributeError:
                    joint_damping = torch.tensor(0.0, device=self.device)
                joint_axis = torch.tensor(joint.axis, device=self.device).reshape(1, 3)

        body_params['rot_angles'] = rot_angles
        body_params['trans'] = trans
        body_params['joint_name'] = joint_name
        body_params['joint_type'] = joint_type
        body_params['joint_limits'] = joint_limits
        body_params['joint_damping'] = joint_damping
        body_params['joint_axis'] = joint_axis
        #body_params['collision_mesh'] = link.collision.geometry.mesh.filename
        if link.inertial is not None:
            mass = torch.tensor(link.inertial.mass, device=self.device)
            com = torch.tensor(link.inertial.origin.position, device=self.device).reshape((1, 3))

            inert_mat = torch.zeros((3, 3), device=self.device)
            inert_mat[0, 0] = link.inertial.inertia.ixx
            inert_mat[0, 1] = link.inertial.inertia.ixy
            inert_mat[0, 2] = link.inertial.inertia.ixz
            inert_mat[1, 0] = link.inertial.inertia.ixy
            inert_mat[1, 1] = link.inertial.inertia.iyy
            inert_mat[1, 2] = link.inertial.inertia.iyz
            inert_mat[2, 0] = link.inertial.inertia.ixz
            inert_mat[2, 1] = link.inertial.inertia.iyz
            inert_mat[2, 2] = link.inertial.inertia.izz

            inert_mat = inert_mat.unsqueeze(0)
            body_params['mass'] = mass
            body_params['com'] = com
            body_params['inertia_mat'] = inert_mat
        else:
            body_params['mass'] = None
            body_params['com'] = None
            body_params['inertia_mat'] = None
            print("no dynamics information for link: {}".format(link.name))

        return body_params
