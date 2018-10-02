import torch
import numpy as np
import argparse
import os.path

from vin.nets_2d import VIN, HVIN, Abstraction_VIN_2D
from vin.nets_2d import _get_path as _get_path_2d
from vin.net_3d import Abstraction_VIN_3D
from vin.net_3d import _get_path as _get_path_3d
from vin.dataloader import GridDataset_2d, GridDataset_3d
from utils import visualize_2d, visualize_3d, get_path_length

# plot 2D grid world paths
def plot_2d(size, map_indices, path_indices, vin=True, hvin=True, abstraction_vin=True):
    # load evaluation set
    dataset = GridDataset_2d(size, data_type='evaluation', full_paths=True)
    
    # load networks
    if vin:
        vin = VIN(size)
        
        # load network state
        if os.path.isfile('network/%s.pt' % vin.name):
            vin.load_state_dict(torch.load('network/%s.pt' % vin.name))
            vin.to(vin.device)
        else:
            print("VIN was not trained.")
            vin = False
            
    if hvin:
        hvin = HVIN(size)
        
        # load network state
        if os.path.isfile('network/%s.pt' % hvin.name):
            hvin.load_state_dict(torch.load('network/%s.pt' % hvin.name))
            hvin.to(hvin.device)
        else:
            print("HVIN was not trained.")
            hvin = False
    
    if abstraction_vin:
        abstraction_vin = Abstraction_VIN_2D(size)
        
        # load network state
        if os.path.isfile('network/%s.pt' % abstraction_vin.name):
            abstraction_vin.load_state_dict(torch.load('network/%s.pt' % abstraction_vin.name))
            abstraction_vin.to(abstraction_vin.device)
        else:
            print("Abstraction_VIN was not trained.")
            abstraction_vin = False
    
    for map_index, path_index in zip(map_indices, path_indices):
        map = dataset.grids[map_index]      # environment map
        optimal_path = dataset.expert_paths[map_index][path_index]
        start = optimal_path[0]
        goal = optimal_path[-1]
        print('Map index: %d' % map_index)
        print('Path index: %d' % path_index)
        print('Optimal path length:  %f' % get_path_length(optimal_path))
        net_paths = []
        
        # predict paths
        if abstraction_vin:
            abstraction_vin_path, abstraction_vin_success = _get_path_2d(abstraction_vin, dataset, map, map_index, start, goal, 2*len(optimal_path))
            abstraction_vin_path = torch.stack(abstraction_vin_path, dim=0)
            if abstraction_vin_success:
                print('Abstraction_VIN path length:  %f' % get_path_length(abstraction_vin_path))
            else:
                print('Abstraction_VIN: No path found.')
            net_paths.append(abstraction_vin_path)
        else:
            net_paths.append(None)
        
        if vin:           
            vin_path, vin_success = _get_path_2d(vin, dataset, map, map_index, start, goal, 2*len(optimal_path))
            vin_path = torch.stack(vin_path, dim=0)
            if vin_success:
                print('VIN path length:  %f' % get_path_length(vin_path))
            else:
                print('VIN: No path found.')
            net_paths.append(vin_path)
        else:
            net_paths.append(None)
                
        if hvin:
            hvin_path, hvin_success = _get_path_2d(hvin, dataset, map, map_index, start, goal, 2*len(optimal_path))
            hvin_path = torch.stack(hvin_path, dim=0)
            if hvin_success:
                print('HVIN path length:  %f' % get_path_length(hvin_path))
            else:
                print('HVIN: No path found.')
            net_paths.append(hvin_path)
        else:
            net_paths.append(None)

        # plot paths
        visualize_2d(map, goal, optimal_path, net_paths, ['Abstraction_VIN','VIN','HVIN'])

# plot paths for 3D task
def plot_3d(size, map_indices, path_indices):  
    # load network
    abstraction_vin = Abstraction_VIN_3D(size)
    
    # load network state
    if os.path.isfile('network/%s.pt' % abstraction_vin.name):
        abstraction_vin.load_state_dict(torch.load('network/%s.pt' % abstraction_vin.name))
        abstraction_vin.to(abstraction_vin.device)
    else:
        print("Abstraction_VIN was not trained.")
        return
    
    for map_index, path_index in zip(map_indices, path_indices):
        map = dataset.grids[map_index]      # environment map
        optimal_path = dataset.expert_paths[map_index][path_index]
        start = optimal_path[0]
        goal = optimal_path[-1]
        print('Map index: %d' % map_index)
        print('Path index: %d' % path_index)
        print('Optimal path length:  %f' % get_path_length(optimal_path, dim=3))
        
        # predict path
        abstraction_vin_path, abstraction_vin_success = _get_path_3d(abstraction_vin, dataset, map, map_index, start, goal, 2*len(optimal_path))
        abstraction_vin_path = torch.stack(abstraction_vin_path, dim=0)
        if abstraction_vin_success:
            print('Abstraction_VIN path length:  %f' % get_path_length(abstraction_vin_path, dim=2))
        else:
            print('Abstraction_VIN: No path found.')

        # plot paths
        visualize_3d(map, goal, optimal_path, abstraction_vin_path)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=2,
        help='Dimensionality of the grid: either 2 for the 2D grid world task or 3 for 3D locomotion planning.')
    parser.add_argument('--size', type=int, default=32, help='Size of grid world [32, 64]')
    parser.add_argument(
        '--num',
        type=int,
        default=1,
        help='Number of different paths that shall be plotted. (Only if no explicit map/path id is given)')
    parser.add_argument(
        '--map_id',
        type=int,
        default=None,
        help='Index of the map that shall be plotted. If none is given, it will be chosen randomly.')
    parser.add_argument(
        '--path_id',
        type=int,
        default=None,
        help='Index of the path that shall be plotted. If none is given, it will be chosen randomly.')
    param = parser.parse_args()
    
    if param.dim == 2:
        # load evaluation set
        dataset = GridDataset_2d(param.size, data_type='evaluation', full_paths=True)
    elif param.dim == 3:
        # load evaluation set
        dataset = GridDataset_3d(param.size, data_type='evaluation', full_paths=True)

    if param.map_id is not None or param.path_id is not None:
        param.num = 1
    
    if param.map_id is None:
        map_indices = np.random.randint(dataset.num_examples, size=param.num)
    else:
        map_indices = [param.map_id]
    
    if param.path_id is None:
        path_indices = np.random.randint(dataset.num_paths_per_map, size=param.num)
    else:
        path_indices = [param.path_id]
    
    if param.dim == 2:
        plot_2d(param.size, map_indices, path_indices)
    elif param.dim == 3:
        plot_3d(param.size, map_indices, path_indices)
    
