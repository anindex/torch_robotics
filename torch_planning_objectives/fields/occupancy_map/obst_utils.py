from math import ceil
import random
import matplotlib.pyplot as plt

from torch_planning_objectives.fields.primitive_distance_fields import Box, Sphere


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier


def random_center(center_lims):
    return [random.uniform(lim_inf, lim_sup) for (lim_inf, lim_sup) in center_lims]


def random_rect(center_lims, shape_dims):
    """
    Generates an rectangular obstacle object, with random location and dimensions.
    """
    center = random_center(center_lims)
    return Box([center], [shape_dims])


def random_circle(center_lims, radius):
    """
    Generates a circle obstacle object, with random location and dimensions.
    """
    center = random_center(center_lims)
    return Sphere([center], [radius])


def save_map_image(obst_map=None,start_pts=None,goal_pts=None,dir='.'):
    try:
        plt.imshow(obst_map,cmap='gray')
        if start_pts is not None:
            for pt in start_pts: plt.plot(pt[0],pt[1],'.g')
        if goal_pts is not None:
            for pt in goal_pts: plt.plot(pt[0],pt[1],'.r')
        plt.gca().invert_yaxis()
        plt.savefig('{}/obst_map.png'.format(dir))
    except Exception as err:
        print("Error: could not save map.")
        print(err)
    return


def get_obst_preset(
        preset_name,
        obst_width=2,
        map_dim=(10,10),
        num_rand_obst=20,
):
    w = obst_width
    map_length = map_dim[0]
    map_height = map_dim[1]
    if preset_name == 'staggered_3-2-3' :
        obst_params = [[-4., 4., w, w], [0., 4., w, w], [4., 4., w, w],
                          [-6, 0, w, w], [-2, 0, w, w], [2, 0, w, w], [6, 0, w, w],
                        [-4., -4., w, w], [0., -4., w, w], [4., -4., w, w]]

    elif preset_name == 'staggered_4-3-4-3-4' :
        obst_params = [[-6, 6, w, w], [-2., 6, w, w], [2., 6, w, w], [6, 6, w, w],
                        [-4., 3, w, w], [0., 3, w, w], [4., 3, w, w],
                        [-6, 0, w, w], [-2., 0, w, w], [2., 0, w, w], [6, 0, w, w],
                        [-4, -3, w, w], [0., -3, w, w], [4, -3, w, w],
                        [-6, -6, w, w], [-2, -6, w, w], [2, -6, w, w], [6, -6, w, w],]

    elif preset_name == 'grid_3x3' :
        s = 5
        obst_params = [[-s, s, w, w], [0., s, w, w], [s, s, w, w],
                        [-s, 0, w, w], [0, 0, w, w], [s, 0, w, w],
                        [-s, -s, w, w], [0., -s, w, w], [s, -s, w, w]]
    elif preset_name == 'grid_4x4' :
        # w = 3
        # w = 2.5
        # w = 2.25
        s = 4
        obst_params = [[-s*3/2, s*3/2, w, w], [-s*1/2, s*3/2, w, w], [s*1/2, s*3/2, w, w], [s*3/2, s*3/2, w, w],
                        [-s*3/2, s/2, w, w], [-s*1/2, s*1/2, w, w], [s*1/2, s*1/2, w, w], [s*3/2, s*1/2, w, w],
                        [-s*3/2, -s*1/2, w, w], [-s*1/2, -s*1/2, w, w], [s*1/2, -s*1/2, w, w], [s*3/2, -s*1/2, w, w],
                        [-s*3/2, -s*3/2, w, w], [-s*1/2, -s*3/2, w, w], [s*1/2, -s*3/2, w, w], [s*3/2, -s*3/2, w, w]]

    elif preset_name == 'grid_6x6' :
        w = obst_width
        # s = 3
        s = 1
        obst_params = [[-s*5/2, s*5/2, w, w], [-s*3/2, s*5/2, w, w], [-s*1/2, s*5/2, w, w], [s*1/2, s*5/2, w, w], [s*3/2, s*5/2, w, w], [s*5/2, s*5/2, w, w],
                       [-s*5/2, s*3/2, w, w], [-s*3/2, s*3/2, w, w], [-s*1/2, s*3/2, w, w], [s*1/2, s*3/2, w, w], [s*3/2, s*3/2, w, w], [s*5/2, s*3/2, w, w],
                       [-s*5/2, s/2, w, w], [-s*3/2, s/2, w, w], [-s*1/2, s*1/2, w, w], [s*1/2, s*1/2, w, w], [s*3/2, s*1/2, w, w],[s*5/2, s*1/2, w, w],
                       [-s*5/2, -s*1/2, w, w], [-s*3/2, -s*1/2, w, w], [-s*1/2, -s*1/2, w, w], [s*1/2, -s*1/2, w, w], [s*3/2, -s*1/2, w, w], [s*5/2, -s*1/2, w, w],
                       [-s*5/2, -s*3/2, w, w], [-s*3/2, -s*3/2, w, w], [-s*1/2, -s*3/2, w, w], [s*1/2, -s*3/2, w, w], [s*3/2, -s*3/2, w, w], [s*5/2, -s*3/2, w, w],
                       [-s*5/2, -s*5/2, w, w], [-s*3/2, -s*5/2, w, w], [-s*1/2, -s*5/2, w, w], [s*1/2, -s*5/2, w, w], [s*3/2, -s*5/2, w, w], [s*5/2, -s*5/2, w, w]]

    elif preset_name == 'maze' :
        b = 6
        # obst_params = [ [-b, b, b, w], [-b, b, w, b],    [0, b, b, w], [0, b, w, b],
        #                 [-b/2, b/2, b, w], [-b/2, b/2, w, b],  [b/2., b/2, b, w], [b/2, b/2, w, b],
        #                [-b, 0., b, w], [-b, 0, w, b],    [0, 0., b, w], [0, 0, w, b],     [b, 0., b, w], [b, 0, w, b],
        #                 [-b/2, -b/2, b, w], [-b/2, -b/2, w, b],   [b/2, -b/2, b, w],[b/2, -b/2, w, b],
        #               [0, -b, b, w], [0, -b, w, b],     [b, -b, b, w], [b, -b, w, b],
        # ]

        obst_params = [ [-b, b, b/2, w], [-b, b, w, b/2],    [0, b, b/2, w], [0, b, w, b/2],   [b, b, b/2, w], [b, b, w, b/2],
                        [-b/2, b/2, b/2, w], [-b/2, b/2, w, b/2],  [b/2., b/2, b/2, w], [b/2, b/2, w, b/2],
                       [-b, 0., b/2, w], [-b, 0, w, b/2],    [0, 0., b/2, w], [0, 0, w, b/2],     [b, 0., b/2, w], [b, 0, w, b/2],
                        [-b/2, -b/2, b/2, w], [-b/2, -b/2, w, b/2],   [b/2, -b/2, b/2, w], [b/2, -b/2, w, b/2],
                      [-b, -b, b/2, w], [-b, -b, w, b/2],   [0, -b, b/2, w], [0, -b, w, b/2],     [b, -b, b/2, w], [b, -b, w, b/2]
        ]

    elif preset_name == 'single_centered' :
        obst_params = [[-5, 0, w, w]]

    elif preset_name == 'rand_halton':
        # num_obst = 22
        # num_obst = 50
        # num_obst = 10
        num_obst = num_rand_obst
        import ghalton
        sequencer = ghalton.Halton(2)
        obst_params = sequencer.get(num_obst)
        for obst in obst_params:
            obst[0] = round_up(obst[0]*map_length - map_length / 2, 0)
            obst[1] = round_up(obst[1]*map_height - map_height / 2, 0)
            # w_rand = randint(1,2)
            # rand_ind = randint(0,1)
            dims = [w, w]
            # dims[rand_ind] = w_rand
            obst += dims
        # obst_params += [[-2, 2, 1, 1], [0, 1, 1, 1]]
        # obst_params += [[-1, 1.5, 2, 1],[0, 0, 1, 2]]
        # obst_params += [[2.5, 8, 2, 1],[5, 6, 1, 2]]

    elif preset_name == 'rand_sobol':
        # num_obst = 28
        # map_width = 18
        num_obst = 15
        import sobol_seq
        obst_params = sobol_seq.i4_sobol_generate(2, num_obst).tolist()
        for obst in obst_params:
            obst[0] = round_up(obst[0]*map_length - map_length / 2, 0)
            obst[1] = round_up(obst[1]*map_height - map_height / 2, 0)
            # obst[0] = round(obst[0]*map_width - map_width / 2, 0)
            # obst[1] = round(obst[1]*map_width - map_width / 2, 0)
            # w_rand = randint(1,2)
            # rand_ind = randint(0,1)
            dims = [w, w]
            # dims[rand_ind] = w_rand
            obst += dims
        obst_params += [[-1, 1.5, 2, 1],[0, 0, 1, 2]]
        # obst_params += [[2.5, 8, 2, 1],[5, 6, 1, 2]]

    elif preset_name == 'rand_mix':
        num_obst = 12
        import sobol_seq
        obst_sobol = sobol_seq.i4_sobol_generate(2, num_obst).tolist()
        for obst in obst_sobol:
            obst[0] = round_up(obst[0]*map_length - map_length / 2, 0)
            obst[1] = round_up(obst[1]*map_height - map_height / 2, 0)
            # obst[0] = round(obst[0]*map_width - map_width / 2, 0)
            # obst[1] = round(obst[1]*map_width - map_width / 2, 0)
            # w_rand = randint(1,2)
            # rand_ind = randint(0,1)
            dims = [w, w]
            # dims[rand_ind] = w_rand
            obst += dims

        num_obst = 8
        import ghalton
        sequencer = ghalton.Halton(2)
        obst_halton = sequencer.get(num_obst)
        for obst in obst_halton:
            obst[0] = round_up(obst[0]*map_length - map_length / 2, 0)
            obst[1] = round_up(obst[1]*map_height - map_height / 2, 0)
            # w_rand = randint(1,2)
            # rand_ind = randint(0,1)
            dims = [w, w]
            # dims[rand_ind] = w_rand
            obst += dims

        obst_params = obst_sobol + obst_halton

        obst_params += [[-1, 5, w, w]]
        obst_params += [[4, 2, w, w]]
        obst_params += [[0, 8, w, w]]
        obst_params += [[6, -1, w, w]]
    else:
        raise IOError('Obstacle preset not supported: ', preset_name)
    return obst_params
