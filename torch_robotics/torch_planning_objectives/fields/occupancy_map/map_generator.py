# MIT License
# Copyright (c) 2022 An Thai Le, João Carvalho

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# **********************************************************************
# The first version of some files were licensed as
# "Original Source License" (see below). Several enhancements and bug fixes
# were done by An Thai Le, João Carvalho since obtaining the first version.



# Original Source License:

# MIT License

# Copyright (c) 2022 Sasha Lambert

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch

from torch_planning_objectives.fields.obst_map import ObstacleMap
from torch_planning_objectives.fields.shape_distance_fields import MultiSphere
from torch_planning_objectives.fields.occupancy_map.obst_utils import random_rect, random_circle
import copy

import matplotlib.pyplot as plt



def build_obstacle_map(
        map_dim=(10, 10),
        cell_size=1.,
        obst_list=[],
        tensor_args=None,
        **kwargs
):
    ## Make occupancy grid
    obst_map = ObstacleMap(map_dim, cell_size, tensor_args=tensor_args)
    for obst in obst_list:
        obst.add_to_occupancy_map(obst_map)

    obst_map.convert_map()

    return obst_map


def generate_obstacle_map(
        map_dim=(10, 10),
        obst_list=[],
        cell_size=1.,
        random_gen=False,
        num_obst=0,
        rand_limits=None,
        rand_rect_shape=[2, 2],
        rand_circle_radius=1,
        tensor_args=None,
):

    """
    Args
    ---
    map_dim : (int,int)
        2-D or 3-D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y, z] coordinates. Origin is in the center.
        ** Dimensions must be an even number. **
    cell_sz : float
        size of each square map cell
    obst_list : [(cx_i, cy_i, width, height)]
        List of obstacle param tuples
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    random_gen : bool
        Specify whether to generate random obstacles. Will first generate obstacles provided by obst_list,
        then add random obstacles until number specified by num_obst.
    num_obst : int
        Total number of obstacles
    rand_limit: [[float, float],[float, float]]
        List defining x-y sampling bounds [[x_min, x_max], [y_min, y_max]]
    rand_shape: [float, float]
        Shape [width, height] of randomly generated obstacles.
    """
    ## Make occupancy grid
    obst_map = ObstacleMap(map_dim, cell_size, tensor_args=tensor_args)
    num_fixed = len(obst_list)
    for obst in obst_list:
        obst.add_to_occupancy_map(obst_map)

    ## Add random obstacles
    obst_list = copy.deepcopy(obst_list)
    if random_gen:
        assert num_fixed <= num_obst, \
            "Total number of obstacles must be >= number specified in obst_list"
        radius = rand_circle_radius
        for _ in range(num_obst - num_fixed):
            num_attempts = 0
            max_attempts = 200
            while num_attempts <= max_attempts:
                if np.random.choice(2):
                    obst = random_rect(rand_limits, rand_rect_shape)
                else:
                    obst = random_circle(rand_limits, radius)

                # Check validity of new obstacle
                # Do not overlap obstacles
                valid = obst.obstacle_collision_check(obst_map)

                if valid:
                    # Add to Map
                    obst.add_to_occupancy_map(obst_map)
                    # Add to list
                    obst_list.append(obst)
                    break

                if num_attempts == max_attempts:
                    print("Obstacle generation: Max. number of attempts reached. ")
                    print("Total num. obstacles: {}.  Num. random obstacles: {}.\n"
                          .format( len(obst_list), len(obst_list) - num_fixed))

                num_attempts += 1

    obst_map.convert_map()

    return obst_map, obst_list


def generate_circle_map(
        map_dim=(10, 10),
        obst_list=[],
        cell_size=1.,
        random_gen=False,
        num_obst=0,
        rand_xy_limits=None,
        rand_circle_radius=[1., 2.],
        map_type=None,
        tensor_args=None,
):

    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates. Origin is in the center.
        ** Dimensions must be an even number. **
    cell_sz : float
        size of each square map cell
    obst_list : [(cx_i, cy_i, width, height)]
        List of obstacle param tuples
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    random_gen : bool
        Specify whether to generate random obstacles. Will first generate obstacles provided by obst_list,
        then add random obstacles until number specified by num_obst.
    num_obst : int
        Total number of obstacles
    rand_limit: [[float, float],[float, float]]
        List defining x-y sampling bounds [[x_min, x_max], [y_min, y_max]]
    rand_shape: [float, float]
        Shape [width, height] of randomly generated obstacles.
    """
    ## Make occupancy grid
    obst_map = ObstacleMap(map_dim, cell_size, tensor_args=tensor_args)
    num_fixed = len(obst_list)
    for obst in obst_list:
        obst._add_to_map(obst_map)

    ## Add random obstacles
    obst_list = copy.deepcopy(obst_list)
    if random_gen:
        assert num_fixed <= num_obst, "Total number of obstacles must be greater than or equal to number specified in obst_list"
        xlim = rand_xy_limits[0]
        ylim = rand_xy_limits[1]
        radius_range = rand_circle_radius
        for _ in range(num_obst - num_fixed):
            num_attempts = 0
            max_attempts = 25
            while num_attempts <= max_attempts:
                radius = np.random.uniform(radius_range[0], radius_range[1])
                obst = random_circle(xlim, ylim, radius)

                # Check validity of new obstacle
                # Do not overlap obstacles
                valid = obst._obstacle_collision_check(obst_map)

                if valid:
                    # Add to Map
                    obst._add_to_map(obst_map)
                    # Add to list
                    obst_list.append(obst)
                    break

                if num_attempts == max_attempts:
                    print("Obstacle generation: Max. number of attempts reached. ")
                    print("Total num. obstacles: {}.  Num. random obstacles: {}.\n"
                          .format( len(obst_list), len(obst_list) - num_fixed))

                num_attempts += 1

    obst_map.convert_map()

    ## Fit mapping model
    if map_type == 'direct':
        return obst_map, obst_list
    else:
        raise IOError('Map type "{}" not recognized'.format(map_type))



def get_sphere_field_from_list(obst_list, field_type='rbf', tensor_args=None):
    """
    Args
    ---
    obst_list : [Obstacle]
        List of Obstacle objects
    """
    field = MultiSphere(obst_type=field_type, tensor_args=tensor_args)
    centers = []
    radii = []
    for obst in obst_list:
        centers.append([obst.center_x, obst.center_y])
        if isinstance(obst, ObstacleBox):
            radii.append(obst.width / 2.)  # NOTE: Assumes width == height
        elif isinstance(obst, ObstacleSphere):
            radii.append(obst.radius)
        else:
            raise IOError('Obstacle type "{}" not recognized'.format(type(obst)))
    field.set_obst(centers, radii)
    return field


if __name__ == "__main__":
    from experiment_launcher.utils import fix_random_seed
    fix_random_seed(1)

    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)

    obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]
    tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
    obst_map, _ = generate_obstacle_map(
        map_dim, obst_list, cell_size,
        random_gen=True,
        # random_gen=False,
        num_obst=20,
        rand_limits=[[-9, 9], [-9, 9]],
        rand_rect_shape=[2, 2],
        rand_circle_radius=1,
        tensor_args=tensor_args
    )

    fig, ax = plt.subplots()

    obst_map.plot(ax)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    traj_y = torch.linspace(-map_dim[1]/2., map_dim[1]/2., 20)
    traj_x = torch.zeros_like(traj_y)
    X = torch.cat((traj_x.unsqueeze(1), traj_y.unsqueeze(1)), dim=1)
    cost = obst_map.get_collisions(X)
    print(cost)



