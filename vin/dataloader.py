import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Data for 2D grid world task
# Initialization parameters:    -size: map size
#                               -data_type: one from ['training','evaluation','validation']
#                               -full_paths: If true, use full expert path, otherwise use randomly chosen sub-path
class GridDataset_2d(Dataset):
    def __init__(self, size, data_type='training', full_paths=False):
        if data_type=='training':
            file_path = 'data_sets/2D/trainingset_'+str(size)+'.pt'
        elif data_type=='evaluation':
            file_path = 'data_sets/2D/evaluationset_'+str(size)+'.pt'
        elif data_type=='validation':
            file_path = 'data_sets/2D/validationset_'+str(size)+'.pt'
        else:
            print('Unknown dataset type.')
        
        file = torch.load(file_path)
        self.grids = file['inputs']             # occupancy maps
        self.expert_paths = file['paths']       # expert paths, where each is a list of grid cell positions
        self.expert_actions = file['actions']   # corresponding lists of action labels
        self.size = size                        # size of grid world
        self.num_examples = len(self.grids)
        self.centre = torch.tensor([self.size//2, self.size//2])
        self.num_paths_per_map = len(self.expert_actions[0])
        self.full_paths=full_paths              # bool: If true, use complete expert path
                                                #       if false, randomly choose sub-path at each iteration
        
        max_path_length = 0
        for path_list in self.expert_paths:
            for path in path_list:
                if len(path)> max_path_length:
                    max_path_length = len(path)
        self.max_path_length = max_path_length

    def __getitem__(self, index):
        map_index = index // self.num_paths_per_map
        path_index = index % self.num_paths_per_map
        
        # expert path and corresponding action sequence for current index
        rollout_path = self.expert_paths[map_index][path_index]
        rollout_actions = self.expert_actions[map_index][path_index]
        
        sampling_max_horizon = len(rollout_actions)


        if self.full_paths:
            t1, t2 = 0, len(rollout_actions)-1
        else:
            if sampling_max_horizon > 1:
                # extract sub-path:
                # sample length first
                length = torch.randint(1, sampling_max_horizon, (1,)).int().item()
                
                # sample start index from path
                t1 = torch.randint(0, sampling_max_horizon-length,(1,)).int().item()
                # compute goal index
                t2 = t1 + length
            else:
                t1, t2 = 0, len(rollout_actions)-1
        
        # get start and goal position
        pos_t1 = rollout_path[t1]
        pos_t2 = rollout_path[t2]
        
        label = rollout_actions[t1]
    
        occ_map, goal_map = self.get_inputs((map_index, pos_t1, pos_t2))
        return {'environment': occ_map, 'goal': goal_map, 'label': label}
    
    def __len__(self):
        return self.num_examples*self.num_paths_per_map
    
    def get_inputs(self, parameters):
        map_index, pos_t1, pos_t2 = parameters
        rollout_img = self.grids[map_index]
        
        # get occupancy map by cropping patch from full environment map around start position
        occ_map = rollout_img[pos_t1[0] - self.size//2:pos_t1[0]+self.size//2,pos_t1[1] - self.size//2:pos_t1[1]+self.size//2]
        
        # get goal map (all cells contain '0' except for goal cell, which contains '1')
        goal_map = torch.FloatTensor(self.size, self.size).fill_(0)
        local_pos_t2 = pos_t2[0:2] - pos_t1[0:2] + self.centre
        if local_pos_t2[0].item() >= 0 and local_pos_t2[0].item() < self.size and local_pos_t2[1].item() >= 0 and local_pos_t2[1].item() < self.size:
            goal_map[local_pos_t2[0], local_pos_t2[1]] = 1
        
        return occ_map.unsqueeze(0), goal_map.unsqueeze(0)

# Data for 3D locomotion planning task
# Initialization parameters:    -size: map size
#                               -data_type: one from ['training','evaluation','validation']
#                               -full_paths: If true, use full expert path, otherwise use randomly chosen sub-path
#                               -num_orientations: number of discrete orientations
class GridDataset_3d(Dataset):
    def __init__(self, size, data_type='training', full_paths=False, num_orientations=16):
        if data_type=='training':
            file_path = 'data_sets/3D/trainingset_'+str(size)+'.pt'
        elif data_type=='evaluation':
            file_path = 'data_sets/3D/evaluationset_'+str(size)+'.pt'
        elif data_type=='validation':
            file_path = 'data_sets/3D/validationset_'+str(size)+'.pt'
        else:
            print('Unknown dataset type.')
    
        file = torch.load(file_path)
        self.grids = file['inputs']             # occupancy maps
        self.expert_paths = file['paths']       # expert paths (each is a list of grid cell positions)
        self.expert_actions = file['actions']   # corresponding sequences of action labels
        self.size = size                    # size of grid world
        self.num_orientations = num_orientations
        self.num_examples = len(self.grids)
        self.centre = torch.tensor([self.size//2, self.size//2])
        self.num_paths_per_map = len(self.expert_actions[0])
        self.full_paths = full_paths            # bool: If true, use complete expert path
                                                #       if false, randomly choose sub-path at each iteration

        max_path_length = 0
        for path_list in self.expert_paths:
            for path in path_list:
                if len(path)> max_path_length:
                    max_path_length = len(path)
        self.max_path_length = max_path_length  

    def __getitem__(self, index):
        map_index = index // self.num_paths_per_map
        path_index = index % self.num_paths_per_map
        
        # expert path and corresponding action sequence for current index
        rollout_actions = self.expert_actions[map_index][path_index]
        rollout_path = self.expert_paths[map_index][path_index]
        
        sampling_max_horizon = len(rollout_actions)

        if self.full_paths:
            t1, t2 = 0, len(rollout_actions)-1
        else:
            if sampling_max_horizon > 1:
                # extract sub-path:
                # sample length first
                length = torch.randint(1, sampling_max_horizon, (1,)).int().item()
                
                # sample start index from path
                t1 = torch.randint(0, sampling_max_horizon-length,(1,)).int().item()
                # compute goal index
                t2 = t1 + length
            else:
                t1, t2 = 0, len(rollout_actions)-1
        
        # get start and goal pose
        pos_t1 = rollout_path[t1]
        pos_t2 = rollout_path[t2]
        orientation = pos_t1[2]
        
        label = rollout_actions[t1]
        
        occ_map, goal_map = self.get_inputs((map_index, pos_t1, pos_t2))
        return {'environment': occ_map, 'goal': goal_map, 'label': label, 'start orientation':orientation}
    
    def __len__(self):
        return self.num_examples*self.num_paths_per_map
    
    def get_inputs(self, parameters):
        map_index, pos_t1, pos_t2 = parameters
        rollout_img = self.grids[map_index]
        
        # get occupancy map by cropping patch from full environment map around start position
        occ_map = rollout_img[pos_t1[0] - self.size//2:pos_t1[0]+self.size//2,pos_t1[1] - self.size//2:pos_t1[1]+self.size//2]
        
        # get goal map (all cells contain '0' except for goal cell, which contains the index of the discrete goal orientation (from 1 to 16)
        goal_map = torch.FloatTensor(self.size, self.size).fill_(0)
        goal_orientation = pos_t2[2]
        local_pos_t2 = pos_t2[0:2] - pos_t1[0:2] + self.centre
        if local_pos_t2[0].item() >= 0 and local_pos_t2[0].item() < self.size and local_pos_t2[1].item() >= 0 and local_pos_t2[1].item() < self.size:
            goal_map[local_pos_t2[0], local_pos_t2[1]] = goal_orientation+1
        
        return occ_map.unsqueeze(0), goal_map.unsqueeze(0)

