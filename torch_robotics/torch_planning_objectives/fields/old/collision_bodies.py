from torch_planning_objectives.fields.distance_fields import SphereDistanceField


class PandaSphereDistanceField(SphereDistanceField):

    def __init__(self, batch_size=1, **kwargs):
        robot_collision_params = {'link_objs': ['panda_link2',
                                                'panda_link3',
                                                'panda_link4',
                                                'panda_link5',
                                                'panda_link6',
                                                'panda_hand'],
                                  'collision_spheres': 'panda/panda_sphere_config.yaml'}

        super(PandaSphereDistanceField, self).__init__(robot_collision_params, batch_size=batch_size, **kwargs)


class TiagoSphereDistanceField(SphereDistanceField):

    def __init__(self, batch_size=1, **kwargs):
        robot_collision_params = {'link_objs': ['torso_fixed_link',
                                                'arm_left_1_link', 'arm_left_2_link', 'arm_left_3_link', 'arm_left_4_link', 'arm_left_5_link', 'arm_left_6_link',
                                                'arm_right_1_link', 'arm_right_2_link', 'arm_right_3_link', 'arm_right_4_link', 'arm_right_5_link', 'arm_right_6_link'],
                                  'collision_spheres': 'tiago/tiago_sphere_config.yaml'}

        super(TiagoSphereDistanceField, self).__init__(robot_collision_params, batch_size=batch_size, **kwargs)

