import numpy as np
import torch
import torch.nn as nn
import random
import time
from multiprocessing import Pool
import argparse

from astar import astar_planner_2d, astar_planner_3d, dijkstra_planner_2d, dijkstra_planner_3d
from utils import path_to_action_vectors, extract_path

# class for environment representation (containing obstacle map, start and goal pose)
class LabeledExample:
    def __init__(self, size):
        self.size = size                    # grid world size
        self.start = (size // 2, size // 2) # start position (center of map)
        self.map = torch.zeros(size, size)  # obstacle map
        self.number_of_obstacles = 0
        self.goal = ()
        
        # only for 3D motion planning
        self.orientation = 0                # start orientation
        self.goal_orientation = 0
        self.num_orientations = 16          # number of discrete orientations
        self.rotation_step_size = 2*np.pi/self.num_orientations # difference between two adjacent discrete orientations
        self.leg_x = 2                      # distance between robot base and wheel (x-coordinate)
        self.leg_y = 2                      # distance between robot base and wheel (y-coordinate)
         
         
    def add_random_rectangular_obstacles(self, number, max_obs_height, max_obs_width):
        while self.number_of_obstacles < number:
            # sample random position
            x = np.random.randint(0, high=self.size)
            y = np.random.randint(0, high=self.size)
             
            # sample random size
            height = np.random.randint(1, high=max_obs_height)
            width = np.random.randint(1, high=max_obs_width)
            
            # ensure that obstacle lies completly within map
            if x+width >= self.size:
                width = self.size-x
            if y+height >= self.size:
                height = self.size-y

            # add obstacle
            self.number_of_obstacles += 1
            self.map[x:x+width, y:y+height] = 1
                

# generates one environment with multiple paths
# Input:    -idx: index of generated map
#           -use_dijkstra: use Dijkstra instead of A* to compute all paths in parallel
# Output: Occupancy map and list of expert paths
def generate_map_with_trajectories(idx, use_dijkstra=True):
    # reset random seed
    np.random.seed()
    # get global parameters
    size, min_number_of_obstacles, max_number_of_obstacles, max_obs_height, max_obs_width, num_paths_per_grid, for_3d = parameter
    
    failed = True
    while failed:
        failed = False
        
        # create empty map of doubled size (to allow cropping patches of the desired size for each position on the expert paths)
        example = LabeledExample(2*size)
        path_list = []
        action_list = []
        
        # add random obstacles
        number_of_obstacles = np.random.randint(4*min_number_of_obstacles, high=4*max_number_of_obstacles)
        example.add_random_rectangular_obstacles(number_of_obstacles, max_obs_height, max_obs_width)       
        
        # set start position at the center of the map
        example.start = (size,size)
        # make sure that the start position does not contain any obstacles
        if for_3d:
            example.map[size-example.leg_x:size+example.leg_x+1,size-example.leg_y:size+example.leg_y+1] = 0
        else:
            example.map[size,size] = 0
        
        num_paths = 0
        counter = 0
        # Use Dijkstra
        if use_dijkstra:
            if for_3d:
                # sample start orientation
                example.orientation = np.random.randint(0,high=example.num_orientations)
                distances, predecessors = dijkstra_planner_3d(example)
            else:
                distances, predecessors = dijkstra_planner_2d(example)

            while num_paths < num_paths_per_grid:
                counter += 1
                # sample random goal state
                x = np.random.randint(size//2+example.leg_x, high= size//2 + size-example.leg_x)
                y = np.random.randint(size//2+example.leg_y, high= size//2 + size-example.leg_y)
                if for_3d:
                    # sample random goal orientation
                    theta = np.random.randint(0,high=example.num_orientations)
  
                    # check if valid path exists:
                    if distances[x,y,theta] != float('Inf'):
                        # get optimal path
                        optimal_path = list(reversed(extract_path((size,size,example.orientation),(x,y,theta),predecessors)))
                        # only allow paths with at least two different poses
                        if len(optimal_path) <= 1:
                            # if not enough paths were generated: start again with empty map
                            if counter > 20:
                                failed = True
                                break
                            continue
                        optimal_path = torch.stack(optimal_path, dim=0)
                        action_sequence = path_to_action_vectors(optimal_path, dim=3)
                        action_sequence = torch.stack(action_sequence, dim=0).float()
                        path_list.append(optimal_path)
                        action_list.append(action_sequence)
                        num_paths +=1
                    else:
                        # if not enough paths were generated: start again with empty map
                        if counter > 20:
                            failed = True
                            break
                else:
                    # check if valid path exists:
                    if distances[x,y] != float('Inf'):
                        # get optimal path
                        optimal_path = list(reversed(extract_path((size,size),(x,y),predecessors, dim=2)))
                        if len(optimal_path) <= 1:
                            # if not enough paths were generated: start again with empty map
                            if counter > 20:
                                failed = True
                                break
                            continue
                        optimal_path = torch.stack(optimal_path, dim=0)
                        action_sequence = path_to_action_vectors(optimal_path, dim=2)
                        action_sequence = torch.stack(action_sequence, dim=0).float()
                        path_list.append(optimal_path)
                        action_list.append(action_sequence)
                        num_paths +=1
                    else:
                        # if not enough paths were generated: start again with empty map
                        if counter > 20:
                            failed = True
                            break
        # Use A* planner
        else:
            while num_paths < num_paths_per_grid:
                counter += 1
                # add random goal
                x = np.random.randint(size//2+example.leg_x, high= size//2 + size-example.leg_x)
                y = np.random.randint(size//2+example.leg_y, high= size//2 + size-example.leg_y)
                example.goal = (x,y)
                # ensure that goal position is free
                example.map[x,y] = 0
                if for_3d:
                    # sample random start and goal orientations
                    example.orientation = np.random.randint(0,high=example.num_orientations)
                    example.goal_orientation = np.random.randint(0,high=example.num_orientations)
                    optimal_path = list(reversed(astar_planner_3d(example)))
                else:
                    optimal_path = list(reversed(astar_planner_2d(example, pos_as_tensor=True)))
                if len(optimal_path) <= 1:
                    if counter > 20:
                        failed = True
                        break
                    continue
                
                optimal_path = torch.stack(optimal_path, dim=0)
                
                if for_3d:
                    action_sequence = path_to_action_vectors(optimal_path, dim=3)
                    action_sequence = torch.stack(action_sequence, dim=0).float()
                else:
                    action_sequence = path_to_action_vectors(optimal_path, dim=2)
                    action_sequence = torch.stack(action_sequence, dim=0).float()
                
                path_list.append(optimal_path)
                action_list.append(action_sequence)
                num_paths +=1
    
    return example.map, path_list, action_list

def generate_data(number_of_examples, size, min_number_of_obstacles, max_number_of_obstacles, max_obs_height=2, max_obs_width=2, num_path_per_grid=7, for_3d=False, data='training', num_workers=4):
    if data=='training':
        print('Generating Training Data')
    elif data=='validation':
        print('Generating Validation Data')
    elif data=='evaluation':
        print('Generating Evaluation Data')
    print('Workers: ', num_workers)

    # make global parameter available for each worker
    global parameter
    parameter = (size, min_number_of_obstacles, max_number_of_obstacles, max_obs_height, max_obs_width, num_path_per_grid, for_3d)
    
    inputs = []
    paths = []
    actions = []
    
        
    pool = Pool(processes=num_workers)
    maps = pool.map(generate_map_with_trajectories, range(number_of_examples))
    pool.close()
    
    for map, path_list, action_list in maps:
        inputs.append(map)
        paths.append(path_list)
        actions.append(action_list)
    
    # save to file
    if data=='training':
        if for_3d:
            file_path = 'data_sets/3D/trainingset_'+str(size)+'.pt'
        else:
            file_path = 'data_sets/2D/trainingset_'+str(size)+'.pt'
    elif data=='validation':
        if for_3d:
            file_path = 'data_sets/3D/validationset_'+str(size)+'.pt'
        else:
            file_path = 'data_sets/2D/validationset_'+str(size)+'.pt'
    elif data=='evaluation':
        if for_3d:
            file_path = 'data_sets/3D/evaluationset_'+str(size)+'.pt'
        else:
            file_path = 'data_sets/2D/evaluationset_'+str(size)+'.pt'
    
    torch.save({'inputs':inputs, 'paths':paths, 'actions':actions}, file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=2,
        help='Dimensionality of the grid: either 2 for the 2D grid world task or 3 for 3D locomotion planning.')
    parser.add_argument('--size', type=int, default=32, help='Size of grid world')
    parser.add_argument(
        '--max_obs_num',
        type=int,
        help='Maximum number of obstacles.')
    parser.add_argument(
        '--min_obs_num',
        type=int,
        help='Minimum number of obstacles.')
    parser.add_argument(
        '--max_obs_height',
        type=int,
        help='Maximum height of obstacles.')
    parser.add_argument(
        '--max_obs_width',
        type=int,
        help='Maximum width of obstacles.')
    parser.add_argument(
        '--paths_per_grid',
        type=int,
        default=7,
        help='Number of expert paths for each grid.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for parallel grid creation.')
    parser.add_argument(
        '--type',
        type=str,
        default='all',
        help='Type of dataset. One of [training, validation, evaluation, all]. Default: all.')
    parser.add_argument(
        '--num_grids',
        type=int,
        help='Number of different grid worlds to be created. Only if --type is set to all, default values will be used:5000 for training and 715 each for validation and evaluation.')
    param = parser.parse_args()
    
    # set size/dim dependent default values
    if param.max_obs_num is None:
        if param.dim == 2:
            param.max_obs_num = 144
        else:
            param.max_obs_num = 21
            
    if param.min_obs_num is None:
        if param.dim == 2:
            param.min_obs_num = 71
        else:
            param.min_obs_num = 6
            
    if param.max_obs_height is None:
        if param.dim == 2:
            param.max_obs_height = 3*param.size//32
        else:
            param.max_obs_height = 5*param.size//32
            
    if param.max_obs_width is None:
        param.max_obs_width = param.max_obs_height
        
    if param.num_grids is None:
        if param.type == 'training':
            param.num_grids = 5000
        else:
            param.num_grids = 715
    
    if param.dim==3:
        for_3d = True
    else:
        for_3d = False
        
    # generate data
    if param.type == 'training':
        generate_data(param.num_grids, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='training', num_workers=param.num_workers)
    elif param.type == 'validation':
        generate_data(param.num_grids, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='validation', num_workers=param.num_workers)
    elif param.type == 'evaluation':
        generate_data(param.num_grids, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='evaluation', num_workers=param.num_workers)
    elif param.type == 'all':
        generate_data(5000, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='training', num_workers=param.num_workers)
        generate_data(715, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='validation', num_workers=param.num_workers)
        generate_data(715, param.size, param.min_obs_num, param.max_obs_num, param.max_obs_height, param.max_obs_width, param.paths_per_grid, for_3d=for_3d, data='evaluation', num_workers=param.num_workers)
    else:
        print('Invalid dataset type.')
        
