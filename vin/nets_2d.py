# Models for 2D grid world task

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool

from dataloader import GridDataset_2d
from utils import get_action, get_path_length


k_values = {4:5, 8:10, 16:20 , 28:36, 32:40, 64:80 , 128:160}

# Value Iteration Network
class VIN(nn.Module):
    def __init__(self, size, k=None, device=None, name=None, hidden_features = 150, num_actions = 9):
        super(VIN, self).__init__() 
        self.size = size            # grid world size
        if name is None:
            self.name = 'VIN_'+str(size)
        else:
            self.name=name
        print("Network name: ", self.name)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k = k or k_values[size]        # number of iterations within VI module
        self.num_actions = num_actions
        
        # learn reward map from input
        self.h = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
        # value iteration module
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.w = Parameter(
            torch.zeros(num_actions, 1, 3, 3), requires_grad=True)
        
        # reactive policy (maps state values to action probabilities)
        self.fc = nn.Linear(in_features=num_actions, out_features=9, bias=False)

    def forward(self, occ_map, goal_map):
        # stack occupancy and goal map
        input_maps = torch.cat([occ_map, goal_map], dim=1)
        
        # generate reward map r
        h = self.h(input_maps)
        r = self.r(h)
        
        # value iteration
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, self.k - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)
        
        # one-step look-ahead
        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=1)
        
        # extract state values from neighbors of start state
        S1 = torch.tensor([[input_maps.size()[2]//2]]).to(self.device)
        S2 = torch.tensor([[input_maps.size()[3]//2]]).to(self.device)
        
        slice_s1 = S1.long().expand(self.size, 1, self.num_actions, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)
        slice_s2 = S2.long().expand(1, self.num_actions, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)
        
        # map state values to action probabilities
        logits = self.fc(q_out)
        return logits
    
    def train(self, num_iterations=1, batch_size=128, validation_step=20, lr= 0.001, plot_curves=False, print_stat=True):
        return _train(self, num_iterations=num_iterations, batch_size=batch_size, validation_step=validation_step, lr=lr, plot_curves=plot_curves, print_stat=print_stat)
    
    # compute next-step accuracy
    def test(self, batch_size=128, validation=True, full_length=True):
        return _test(self, batch_size=batch_size, validation=validation, full_length=full_length)
    
    # compute success rate for whole paths
    def rollout(net, batch_size=128, validation=True, num_workers=4):
        return _rollout(net, batch_size=batch_size, validation=validation, num_workers=num_workers)

# Hierarchical Value Iteration Network
# (planning on multiple resolutions without compensation lower resolution with additional features)
class HVIN(nn.Module):
    def __init__(self, size, k=None, device=None, name=None, hidden_features = 150, num_actions = 9):
        super(HVIN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.size = size
        if name is None:
            self.name = 'HVIN_'+str(size)
        else:
            self.name=name
        print("Network name: ", self.name)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k3 = k or k_values[size]//4    # number of iteration within VI module at level 3
        self.k2 = 2     # number of iteration within VI module at level 2 (refinement of level 3 values)
        self.k1 = 2     # number of iteration within VI module at level 1 (refinement of level 2 values)
        self.num_actions = num_actions
        
        # Processing on original resolution:
            # generate reward map
        self.h1 = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r1 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
            
            # value iteration
        self.q1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.w1 = Parameter(torch.zeros(num_actions, 1, 3, 3), requires_grad=True)
        
        # Processing on half resolution:
            # generate reward map
        self.h2 = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
            # value iteration
        self.q2 = nn.Conv2d(
            in_channels=1,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.w2 = Parameter(torch.zeros(num_actions, 1, 3, 3), requires_grad=True)
        
        # Processing on quarter of original resolution:
            # generate reward map
        self.h3 = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_features,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r3 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
            # value iteration
        self.q3 = nn.Conv2d(
            in_channels=1,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.w3 = Parameter(torch.zeros(num_actions, 1, 3, 3), requires_grad=True)
        
        # map state values to action probabilities
        self.fc = nn.Linear(in_features=num_actions, out_features=9, bias=False)

    def forward(self, occ_map, goal_map):
        # input maps at full resolution
        level_1 = torch.cat([occ_map, goal_map], dim=1)
        h_1 = self.h1(level_1)
        r_1 = self.r1(h_1)
        
        # down-sample input maps (half resolution)
        level_2 = F.avg_pool2d(level_1, (2,2))
        h_2 = self.h2(level_2)
        r_2 = self.r2(h_2)
        
        # down-sample input maps (quarter of original resolution)
        level_3 = F.avg_pool2d(level_1, (4,4))
        h_3 = self.h3(level_3)
        r_3 = self.r3(h_3)
            
        # value iteration on level 3 (one quarter of resolution)
        q = self.q3(r_3)
        v_3, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, self.k3 - 1):
            q = F.conv2d(
                torch.cat([r_3, v_3], 1),
                torch.cat([self.q3.weight, self.w3], 1),
                stride=1,
                padding=1)
            v_3, _ = torch.max(q, dim=1, keepdim=True)

        
        # value iteration on level 2
        # (refine state values from level 3)
        v_2 = F.upsample(v_3, scale_factor=2, mode='nearest')
        for i in range(0, self.k2):
            q = F.conv2d(
                torch.cat([r_2, v_2], 1),
                torch.cat([self.q2.weight, self.w2], 1),
                stride=1,
                padding=1)
            v_2, _ = torch.max(q, dim=1, keepdim=True)
        
        # value iteration on level 1
        # (refine state values from level 2)
        v_1 = F.upsample(v_2, scale_factor=2, mode='nearest')
        for i in range(0, self.k1):
            q = F.conv2d(
                torch.cat([r_1, v_1], 1),
                torch.cat([self.q1.weight, self.w1], 1),
                stride=1,
                padding=1)
            v_1, _ = torch.max(q, dim=1, keepdim=True)
        
        # one-step look-ahead
        q = F.conv2d(
            torch.cat([r_1, v_1], 1),
            torch.cat([self.q1.weight, self.w1], 1),
            stride=1,
            padding=1)
        
        # extract state values from neighbors of start state
        S1 = torch.tensor([[level_1.size()[2]//2]]).to(self.device)
        S2 = torch.tensor([[level_1.size()[3]//2]]).to(self.device)
        
        slice_s1 = S1.long().expand(self.size, 1, self.num_actions, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)
        slice_s2 = S2.long().expand(1, self.num_actions, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)
        
        # map state values to action probabilities
        logits = self.fc(q_out)
        return logits
    
    def train(self, num_iterations=1, batch_size=128, validation_step=20, lr= 0.001, plot_curves=False, print_stat=True):
        return _train(self, num_iterations=num_iterations, batch_size=batch_size, validation_step=validation_step, lr=lr, plot_curves=plot_curves, print_stat=print_stat)
    
    # compute next-step accuracy
    def test(self, batch_size=128, validation=True, full_length=True):
        return _test(self, batch_size=batch_size, validation=validation, full_length=full_length)
    
    # compute success rate for whole paths
    def rollout(net, batch_size=128, validation=True, num_workers=4):
        return _rollout(net, batch_size=batch_size, validation=validation, num_workers=num_workers)

# Value Iteration Network on multiple levels of abstraction
class Abstraction_VIN_2D(nn.Module):
    def __init__(self, size, k=None, device = None, num_actions = 9, 
                name=None,
                    level_2_features = 2,   # number of features for Level-2 representation
                    level_3_features = 6,   # number of features for Level-3 representation
                    level_1_conv_features = [150],
                    level_1_conv_kernels = [(3,3)],
                    level_1_conv_paddings = [1],
                    level_2_conv_features = [150],
                    level_2_conv_kernels = [(3,3)],
                    level_2_conv_paddings = [1],
                    level_3_conv_features = [150],
                    level_3_conv_kernels = [(3,3)],
                    level_3_conv_paddings = [1]):
        super(Abstraction_VIN_2D, self).__init__()
        self.size = size            # grid world size
        self.size_eff = size//4     # size of each abstraction map
        
        if name is None:
            self.name = 'Abstraction_VIN_2D_'+str(size)
        else:
            self.name=name
        print("Network name: ", self.name)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k = k or int(1.5*k_values[self.size_eff])  # number of iterations within VI module
        self.num_actions = num_actions
        self.level_2_features = level_2_features
        self.level_3_features = level_3_features
        self.num_features = 1 + level_2_features + level_3_features     # overall number of features of reward map (sum over all 3 levels)
        
        # learn abstract representations
        self.learn_level_2 = nn.Conv2d(
            in_channels=1,
            out_channels=level_2_features,
            kernel_size=(2, 2),
            stride=2,
            padding=0,
            bias=False)
        
        self.learn_level_3 = nn.Conv2d(
            in_channels=level_2_features,
            out_channels=level_3_features,
            kernel_size=(2, 2),
            stride=2,
            padding=0,
            bias=False)
        
        # process level 1
        self.abstraction_1_pad = nn.ConstantPad2d(int(0.25*self.size_eff), 0)
        self.level_1_conv = nn.ModuleList()
        
        self.level_1_conv.append(nn.Conv2d(in_channels=2,
                                           out_channels=level_1_conv_features[0],
                                           kernel_size=level_1_conv_kernels[0],
                                           stride=1,
                                           padding=level_1_conv_paddings[0],
                                           bias=True))
        
        for i in range(1, len(level_1_conv_features)):
            self.level_1_conv.append(nn.Conv2d(in_channels=level_1_conv_features[i-1],
                                    out_channels=level_1_conv_features[i],
                                    kernel_size=level_1_conv_kernels[i],
                                    stride=1,
                                    padding=level_1_conv_paddings[i],
                                    bias=True))
        
        # process level 2
        self.abstraction_2_pad = nn.ConstantPad2d(self.size_eff//4, 0)
        self.level_2_conv = nn.ModuleList()
        
        self.level_2_conv.append(nn.Conv2d(in_channels=1+level_2_features+1,
                                           out_channels=level_2_conv_features[0],
                                           kernel_size=level_2_conv_kernels[0],
                                           stride=1,
                                           padding=level_2_conv_paddings[0],
                                           bias=True))
        
        for i in range(1, len(level_2_conv_features)):
            self.level_2_conv.append(nn.Conv2d(in_channels=level_2_conv_features[i-1],
                                    out_channels=level_2_conv_features[i],
                                    kernel_size=level_2_conv_kernels[i],
                                    stride=1,
                                    padding=level_2_conv_paddings[i],
                                    bias=True))
            
        # process level 3
        self.level_3_conv = nn.ModuleList()
        self.level_3_conv.append(nn.Conv2d(in_channels=level_2_features+level_3_features+1,
                                           out_channels=level_3_conv_features[0],
                                           kernel_size=level_3_conv_kernels[0],
                                           stride=1,
                                           padding=level_3_conv_paddings[0],
                                           bias=True))
        
        for i in range(1, len(level_3_conv_features)):
            self.level_3_conv.append(nn.Conv2d(in_channels=level_3_conv_features[i-1],
                                    out_channels=level_3_conv_features[i],
                                    kernel_size=level_3_conv_kernels[i],
                                    stride=1,
                                    padding=level_3_conv_paddings[i],
                                    bias=True))
        
        # generate reward map:
            # Level-1
        self.r1 = nn.Conv2d(
            in_channels=level_1_conv_features[-1],
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
            # Level-2
        self.r2 = nn.Conv2d(
            in_channels=level_2_conv_features[-1],
            out_channels=level_2_features,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
            # Level-3
        self.r3 = nn.Conv2d(
            in_channels=level_3_conv_features[-1],
            out_channels=level_3_features,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        
        # value iteration:    
            # Level-1
        self.q1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=False)
        
            # Level-2
        self.q2 = nn.Conv2d(
            in_channels=2,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=False)
        
            # Level-3
        self.q3 = nn.Conv2d(
            in_channels=6,
            out_channels=num_actions,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            bias=False)

        self.w = Parameter(
            torch.zeros(num_actions, 1, 3, 3), requires_grad=True)
        
        # reactive policy (maps state values to action probabilities)
        self.fc = nn.Linear(in_features=9, out_features=num_actions, bias=False)
        

    def forward(self, occ_map, goal_map):
        # Create abstraction maps:
            # extract Level-1 map
        level_1 = occ_map[:,:,self.size//2 - self.size_eff//2: self.size//2 + self.size_eff//2,self.size//2 - self.size_eff//2: self.size//2 + self.size_eff//2]
        
            # create Level-2 representation
        level_2_full = self.learn_level_2(occ_map)
        
            # extract Level-2 map
        level_2 = level_2_full[:,:,self.size//4 - self.size_eff//2: self.size//4 + self.size_eff//2,self.size//4 - self.size_eff//2: self.size//4 + self.size_eff//2]
        
            # create Level-3 map
        level_3 = self.learn_level_3(level_2_full)
        
        # Create abstract goal maps
        goal_1 = goal_map[:,:,self.size//2 - self.size_eff//2: self.size//2 + self.size_eff//2,self.size//2 - self.size_eff//2: self.size//2 + self.size_eff//2]
        goal_2_full = F.max_pool2d(goal_map, (2,2))
        goal_2 = goal_2_full[:,:,self.size//4 - self.size_eff//2: self.size//4 + self.size_eff//2,self.size//4 - self.size_eff//2: self.size//4 + self.size_eff//2]
        goal_3 = F.max_pool2d(goal_2_full, (2,2))
        
        # Process Level-1:
            # stack with goal map
        abstraction_1 = torch.cat([level_1, goal_1], dim=1)
        
            # generate Level-1 reward map
        for conv in self.level_1_conv:
            abstraction_1 = conv(abstraction_1)
            
        abstraction_1 = self.r1(abstraction_1)
        r1 = abstraction_1        

            # reduce resolution (to fit resolution of Level-2)
        abstraction_1 = F.max_pool2d(abstraction_1, (2,2))
        
            # pad abstraction_1 map (to match size of Level-2 map)
        abstraction_1 = self.abstraction_1_pad(abstraction_1)
        
        # Process Level-2:
            # stack with preprocessed Level-1 map and Level-2 goal map
        abstraction_2 = torch.cat([abstraction_1, level_2, goal_2], dim=1)
        
            # generate Level-2 reward map
        for conv in self.level_2_conv:
            abstraction_2 = conv(abstraction_2)
            
        abstraction_2 = self.r2(abstraction_2)
        r2 = abstraction_2          
        
            # reduce resolution (to fit resolution of Level-3)
        abstraction_2 = F.max_pool2d(abstraction_2, (2,2))
        
            # pad abstraction_2 map (to match Level-3 size)
        abstraction_2 = self.abstraction_2_pad(abstraction_2)
        
        # Process Level-3:
            # stack with preprocessed Level-2 map and Level-3 goal map
        abstraction_3 = torch.cat([abstraction_2, level_3, goal_3], dim=1)
        
            # generate Level-3 reward map
        for conv in self.level_3_conv:
            abstraction_3 = conv(abstraction_3)
        
        r3 = self.r3(abstraction_3)

        # generate reward map
        r = torch.cat([r1,r2,r3], dim=1)
        
        # pad reward map (to get rewards for transitions between different abstraction levels)
        r_pad = F.pad(r,(1,1,1,1), 'constant', 0)
        
        # transitions from high to low abstraction levels:
            # from Level-2 to Level-1
        r_pad[:,0,1:-1,0] = F.interpolate(r[:,1:self.level_2_features+1,self.size_eff//4:-self.size_eff//4,self.size_eff//4-1].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,0,1:-1,-1] = F.interpolate(r[:,1:self.level_2_features+1,self.size_eff//4:-self.size_eff//4,-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,0,0,1:-1] = F.interpolate(r[:,1:self.level_2_features+1,self.size_eff//4-1,self.size_eff//4:-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,0,-1,1:-1] = F.interpolate(r[:,1:self.level_2_features+1,-self.size_eff//4,self.size_eff//4:-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        
        r_pad[:,0,0,0] = r[:,1:3,self.size_eff//4-1,self.size_eff//4-1].mean(1,keepdim=True).squeeze(1)
        r_pad[:,0,0,-1] = r[:,1:3,self.size_eff//4-1,-self.size_eff//4].mean(1,keepdim=True).squeeze(1)
        r_pad[:,0,-1,0] = r[:,1:3,-self.size_eff//4,self.size_eff//4-1].mean(1,keepdim=True).squeeze(1)
        r_pad[:,0,-1,-1] = r[:,1:3,-self.size_eff//4,-self.size_eff//4].mean(1,keepdim=True).squeeze(1)
        
            # from Level-3 to Level-2
        r_pad[:,1,1:-1,0] = F.interpolate(r[:,self.level_2_features+1:,self.size_eff//4:-self.size_eff//4,self.size_eff//4-1].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,1,1:-1,-1] = F.interpolate(r[:,self.level_2_features+1:,self.size_eff//4:-self.size_eff//4,-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,1,0,1:-1] = F.interpolate(r[:,self.level_2_features+1:,self.size_eff//4-1,self.size_eff//4:-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)
        r_pad[:,1,-1,1:-1] = F.interpolate(r[:,self.level_2_features+1:,-self.size_eff//4,self.size_eff//4:-self.size_eff//4].mean(1,keepdim=True), scale_factor = 2).squeeze(1)

        r_pad[:,1,0,0] = r[:,3:,self.size_eff//4-1,self.size_eff//4-1].mean(1,keepdim=True).squeeze(1)
        r_pad[:,1,0,-1] = r[:,3:,self.size_eff//4-1,-self.size_eff//4].mean(1,keepdim=True).squeeze(1)
        r_pad[:,1,-1,0] = r[:,3:,-self.size_eff//4,self.size_eff//4-1].mean(1,keepdim=True).squeeze(1)
        r_pad[:,1,-1,-1] = r[:,3:,-self.size_eff//4,-self.size_eff//4].mean(1,keepdim=True).squeeze(1)
        
            # use same padding values for all Level-2 features
        for feature in range(2,self.level_2_features+1):
            r_pad[:,feature,1:-1,0] = r_pad[:,1,1:-1,0]
            r_pad[:,feature,1:-1,-1] = r_pad[:,1,1:-1,-1]
            r_pad[:,feature,0,1:-1] = r_pad[:,1,0,1:-1]
            r_pad[:,feature,-1,1:-1] = r_pad[:,1,-1,1:-1]
            
            r_pad[:,feature,0,0] = r_pad[:,1,0,0]
            r_pad[:,feature,0,-1] = r_pad[:,1,0,-1]
            r_pad[:,feature,-1,0] = r_pad[:,1,-1,0]
            r_pad[:,feature,-1,-1] = r_pad[:,1,-1,-1]
        
        # value iteration (on each abstraction level in parallel)
        q1 = self.q1(r_pad[:,0,:,:].unsqueeze(1))
        q2 = self.q2(r_pad[:,1:self.level_2_features+1,:,:])
        q3 = self.q3(r_pad[:,self.level_2_features+1:self.num_features,:,:])

        v1, _ = torch.max(q1, dim=1, keepdim=True)
        v2, _ = torch.max(q2, dim=1, keepdim=True)
        v3, _ = torch.max(q3, dim=1, keepdim=True)
        v = torch.cat([v1,v2,v3],1)
        
        for i in range(0, self.k - 1):
            # information flow between levels after each iteration:
                # information flow from low to high abstraction level
                # (replace state value for more abstract cell with maximum state value from
                # the lower-level cells that describe the same area)
            v[:,1,self.size_eff//4:-self.size_eff//4,self.size_eff//4:-self.size_eff//4] = F.max_pool2d(v[:,0,:,:].clone(), (2,2))
            v[:,2,self.size_eff//4:-self.size_eff//4,self.size_eff//4:-self.size_eff//4] = F.max_pool2d(v[:,1,:,:].clone(), (2,2))
            
            
                # information flow from high to low abstraction level
                # (pad lower-level with values from neighboring higher-level cells)
            v_pad = F.pad(v,(1,1,1,1), 'constant', 0)
            v_pad[:,0:2,1:-1,0] = F.interpolate(v[:,1:3,self.size_eff//4:-self.size_eff//4,self.size_eff//4-1], scale_factor = 2)
            v_pad[:,0:2,1:-1,-1] = F.interpolate(v[:,1:3,self.size_eff//4:-self.size_eff//4,-self.size_eff//4], scale_factor = 2)
            v_pad[:,0:2,0,1:-1] = F.interpolate(v[:,1:3,self.size_eff//4-1,self.size_eff//4:-self.size_eff//4], scale_factor = 2)
            v_pad[:,0:2,-1,1:-1] = F.interpolate(v[:,1:3,-self.size_eff//4,self.size_eff//4:-self.size_eff//4], scale_factor = 2)

            v_pad[:,0:2,0,0] = v[:,1:3,self.size_eff//4-1,self.size_eff//4-1]
            v_pad[:,0:2,0,-1] = v[:,1:3,self.size_eff//4-1,-self.size_eff//4]
            v_pad[:,0:2,-1,0] = v[:,1:3,-self.size_eff//4,self.size_eff//4-1]
            v_pad[:,0:2,-1,-1] = v[:,1:3,-self.size_eff//4,-self.size_eff//4]
        
                # Bellman update (on each abstraction level in parallel)
            q1 = F.conv2d(
                torch.cat([r_pad[:,0,:,:].unsqueeze(1), v_pad[:,0,:,:].unsqueeze(1)], 1),
                torch.cat([self.q1.weight, self.w], 1),
                stride=1,
                padding=0)
            q2 = F.conv2d(
                torch.cat([r_pad[:,1:3,:,:], v_pad[:,1,:,:].unsqueeze(1)], 1),
                torch.cat([self.q2.weight, self.w], 1),
                stride=1,
                padding=0)
            q3 = F.conv2d(
                torch.cat([r_pad[:,3:9,:,:], v_pad[:,2,:,:].unsqueeze(1)], 1),
                torch.cat([self.q3.weight, self.w], 1),
                stride=1,
                padding=0)

            v1, _ = torch.max(q1, dim=1, keepdim=True)
            v2, _ = torch.max(q2, dim=1, keepdim=True)
            v3, _ = torch.max(q3, dim=1, keepdim=True)
            v = torch.cat([v1,v2,v3],1)

        # information flow from low to high abstraction level
        v[:,1,self.size_eff//4:-self.size_eff//4,self.size_eff//4:-self.size_eff//4] = F.max_pool2d(v[:,0,:,:].clone(), (2,2))
        v[:,2,self.size_eff//4:-self.size_eff//4,self.size_eff//4:-self.size_eff//4] = F.max_pool2d(v[:,1,:,:].clone(), (2,2))
        
        # information flow from high to low abstraction level
        v_pad = F.pad(v,(1,1,1,1), 'constant', 0)
        v_pad[:,0:2,1:-1,0] = F.interpolate(v[:,1:3,self.size_eff//4:-self.size_eff//4,self.size_eff//4-1], scale_factor = 2)
        v_pad[:,0:2,1:-1,-1] = F.interpolate(v[:,1:3,self.size_eff//4:-self.size_eff//4,-self.size_eff//4], scale_factor = 2)
        v_pad[:,0:2,0,1:-1] = F.interpolate(v[:,1:3,self.size_eff//4-1,self.size_eff//4:-self.size_eff//4], scale_factor = 2)
        v_pad[:,0:2,-1,1:-1] = F.interpolate(v[:,1:3,-self.size_eff//4,self.size_eff//4:-self.size_eff//4], scale_factor = 2)
        
        v_pad[:,0:2,0,0] = v[:,1:3,self.size_eff//4-1,self.size_eff//4-1]
        v_pad[:,0:2,0,-1] = v[:,1:3,self.size_eff//4-1,-self.size_eff//4]
        v_pad[:,0:2,-1,0] = v[:,1:3,-self.size_eff//4,self.size_eff//4-1]
        v_pad[:,0:2,-1,-1] = v[:,1:3,-self.size_eff//4,-self.size_eff//4]
        
        # one-step look-ahead
        q1 = F.conv2d(
            torch.cat([r_pad[:,0,:,:].unsqueeze(1), v_pad[:,0,:,:].unsqueeze(1)], 1),
            torch.cat([self.q1.weight, self.w], 1),
            stride=1,
            padding=0)

        # get state values for neighbors of start state
        q_out = q1[:,:,self.size_eff//2, self.size_eff//2]
        
        # map state values to action probabilities
        logits = self.fc(q_out)
        return logits
    
    def train(self, num_iterations=1, batch_size=128, validation_step=20, lr= 0.001, plot_curves=False, print_stat=True):
        return _train(self, num_iterations=num_iterations, batch_size=batch_size, validation_step=validation_step, lr=lr, plot_curves=plot_curves, print_stat=print_stat)
    
    # compute next-step accuracy
    def test(self, batch_size=128, validation=True, full_length=True):
        return _test(self, batch_size=batch_size, validation=validation, full_length=full_length)
    
    # compute success rate for whole paths
    def rollout(net, batch_size=128, validation=True, num_workers=4):
        return _rollout(net, batch_size=batch_size, validation=validation, num_workers=num_workers)
    
# train the network using RMSprop
def _train(net, num_iterations=1, batch_size=128, validation_step=20, lr= 0.001, plot_curves=False, print_stat=True):
    # load training data
    dataset = GridDataset_2d(net.size, data_type='training', full_paths=False)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr, eps=1e-6)
    error_stat = []
    evaluation_stat = []
    rollout_stat = []
    best_success = 0.

    print('Starting Training.')
    print('Epochs: ', num_iterations)
    print('Batch size: ', batch_size)
    print('Expert demonstrations: ', len(dataset))
    print('Optimizer: ', optimizer)
    
    for epoch in range(num_iterations):
        running_loss = 0.0
        num_batches = 0
        start_time_epoch = time.time()
        
        # run once over all training examples
        for i, data in enumerate(trainloader, 0):
            # reset gradients to zero
            optimizer.zero_grad()
            
            # get training data
            occ_maps, goal_maps, labels = data['environment'].to(net.device), data['goal'].to(net.device), data['label'].to(net.device)
            
            # forward pass
            outputs = net.forward(occ_maps, goal_maps)
            
            # compute training loss
            loss = criterion(outputs,torch.max(labels, 1)[1])
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
                
        duration_epoch = time.time() - start_time_epoch
        
        # track average loss for learning curve
        error_stat.append(running_loss/num_batches)
        
        # test network on validation set
        if epoch % validation_step == validation_step-1:
            accuracy = net.test(batch_size=batch_size, full_length=True, validation=True)
            accuracy_sampled = net.test(batch_size=batch_size, full_length=False, validation=True)
            evaluation_stat.append(accuracy)
            success=net.rollout(batch_size=batch_size, validation=True)
            rollout_stat.append(success)
            print('[epoch %d] loss per batch: %.10f, accuracy (full): %f, accuracy (sampled): %f, success: %f,time: %f' % (epoch + 1, running_loss / num_batches, accuracy, accuracy_sampled, success, duration_epoch))
            
            # save network state which achieves best success rate on validation set
            if success > best_success:
                best_success = success
                torch.save(net.state_dict(), 'network/%s.pt' % net.name)
            
        else:
            print('[epoch %d] loss per batch: %.10f, time: %f' % (epoch + 1, running_loss / num_batches, duration_epoch))

    print('Finished Training.')
    print('')
    
    # get network state with best success rate on validation set
    net.load_state_dict(torch.load('network/%s.pt' % net.name))
    
    # plot learning curves
    if plot_curves:
        plt.figure(0)
        plt.plot(range(len(error_stat)), error_stat)
        plt.savefig('learning_curves/training_loss_%s.png' % net.name)
        
        plt.figure(1)
        plt.plot(range(len(evaluation_stat)), evaluation_stat)
        plt.savefig('learning_curves/accuracy_%s.png' % net.name)
        
        plt.figure(2)
        plt.plot(range(len(rollout_stat)), rollout_stat)
        plt.savefig('learning_curves/success_%s.png' % net.name)
    
    # print training statistics to text file
    if print_stat:
        error_stat = np.array(error_stat)
        epoch_list = np.arange(1,len(error_stat)+1)
        train_data = np.column_stack((epoch_list, error_stat))
        np.savetxt('learning_curves/training_loss_%s.txt' % net.name, train_data, delimiter = " ", fmt=("%d","%f"), header = "Epoch Loss")
        
        epoch_list = np.arange(validation_step, num_iterations+1, validation_step)
        evaluation_stat = np.array(evaluation_stat)
        rollout_stat = np.array(rollout_stat)
        validation_data = np.column_stack((epoch_list, evaluation_stat, rollout_stat))
        np.savetxt('learning_curves/validation_%s.txt' % net.name, validation_data, delimiter = " ", fmt=("%d","%f","%f"), header = "Epoch Accuracy Success")
        
# compute next-step accuracy
def _test(net, batch_size=128, validation=True, full_length=True):
    with torch.no_grad():
        if validation:
            # load validation set
            dataset = GridDataset_2d(net.size, data_type='validation', full_paths=full_length)
        else:
            print('Starting Test.')
            print('Full length: ', full_length)
            
            # load evaluation set
            dataset = GridDataset_2d(net.size, data_type='evaluation', full_paths=full_length)
            
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        num_wrong_actions = 0
        num_batches = 0
        running_loss = 0.
        start_time = time.time()
        
        # iterate once over each example
        for i, data in enumerate(testloader, 0):
            occ_maps, goal_maps, labels= data['environment'].to(net.device), data['goal'].to(net.device), data['label'].to(net.device)             
            output = net.forward(occ_maps, goal_maps)
            num_batches += 1
            
            # count wrong actions
            for j in range(output.size()[0]):
                # get action with highest probability
                action = output[j].max(0)[1]
                action = action.item()
                
                # get expert action
                label = labels[j].argmax(0).item()
                if action != label:
                    num_wrong_actions += 1
            
        duration = time.time() - start_time
        accuracy = 1 - num_wrong_actions/len(dataset)
        
        if not validation:
            print('Size of Test Set: ', len(dataset))
            print('Loss per batch: ', running_loss / num_batches)
            print('Number of wrong actions: ', num_wrong_actions)
            print("Accuracy: ", accuracy)
            print("Time: ", duration)
            print('')
        return accuracy

# unroll single full path (for visualization) 
def _get_path(net, dataset, map, map_index, start_pos, goal_pos, max_number_steps):
    with torch.no_grad():
        success = True
        path = [start_pos]
        pos = start_pos

        for idx in range(max_number_steps):
            # ensure that whole perceptive area lies within grid world
            if pos[0] >= 3*map.size()[0]//4 or pos[0] < map.size()[0]//4 or pos[1] >= 3*map.size()[1]//4 or pos[1] < map.size()[1]//4:
                return (path, False)
            
            # reached goal
            if pos[0] == goal_pos[0] and pos[1] == goal_pos[1]:
                return (path, success)
            
            # check collision
            if map[pos[0],pos[1]] == 1:
                success = False
            
            # get input maps for current position
            occ_map, goal_map = dataset.get_inputs((map_index, pos, goal_pos))
            occ_map, goal_map = occ_map.unsqueeze_(0).to(net.device), goal_map.unsqueeze_(0).to(net.device)
            
            # predict next action
            action_vector = net.forward(occ_map, goal_map)
            action = get_action(action_vector[0], dim = 2)
            
            # update position
            new_pos = pos + action
            path.append(new_pos)
            pos = new_pos
        
        if pos[0] == goal_pos[0] and pos[1] == goal_pos[1] and pos[2] == goal_pos[2]:
            # reached goal
            return (path, success)
        else:
            # did not reach goal
            return (path, False)

# compute success rate for whole paths    
def _rollout(net, batch_size=128, validation=True, num_workers=4):
    with torch.no_grad():
        diff = 0.
        net_length = 0.
        expert_length = 0.
        
        # load dataset and make it available to all workers
        global rollout_data
        if validation:
            rollout_data = GridDataset_2d(net.size, data_type='validation', full_paths=True)
        else:
            rollout_data = GridDataset_2d(net.size, data_type='evaluation', full_paths=True)
        iterations = rollout_data.num_examples
        
        # list of all tasks (describes task through map and path indices)
        open_paths = [(i,j) for i in range(rollout_data.num_examples) for j in range(rollout_data.num_paths_per_map)]
        paths = [[[rollout_data.expert_paths[map_id][path_id][0]] for path_id in range(rollout_data.num_paths_per_map)] for map_id in range(rollout_data.num_examples)]
        success = [[ False for path_id in range(rollout_data.num_paths_per_map)] for map_id in range(rollout_data.num_examples)]
        
        path_length = 0
        if not validation:
            print("Starting Rollout-Test.")
            print("Max expert path length:", rollout_data.max_path_length)
            start_time = time.time()
        
        pool = Pool(processes=num_workers)
        while len(open_paths) != 0 and path_length < 2*rollout_data.max_path_length:
            parameters = []
            # get map indices and current positions for all open paths
            for map_id, path_id in open_paths:
                parameters.append((map_id, paths[map_id][path_id][-1], rollout_data.expert_paths[map_id][path_id][-1]))
                
            # get inputs for all open paths
            inputs = pool.map(_get_inputs, parameters)
            
            path_length += 1
            current_open_task_id = 0
            
            # predict next step for each open path
            for input_batch in batch(inputs,batch_size):
                # unpack inputs
                occ_maps, goal_maps = zip(*input_batch)
                occ_maps, goal_maps = torch.stack(occ_maps, dim=0).to(net.device), torch.stack(goal_maps,dim=0).to(net.device)
                
                # predict next action
                action_vectors = net.forward(occ_maps, goal_maps)
                
                for i in range(action_vectors.size(0)):
                    # update positions and paths
                    map_id, path_id = open_paths[current_open_task_id]
                    action = get_action(action_vectors[i], dim=2)
                    pos = paths[map_id][path_id][-1] + action
                    
                    paths[map_id][path_id].append(pos)
                    goal_pos = rollout_data.expert_paths[map_id][path_id][-1]
                    
                    # reached goal
                    if pos[0] == goal_pos[0] and pos[1] == goal_pos[1]:
                        success[map_id][path_id] = True
                        del open_paths[current_open_task_id]
                        continue
                    
                    # check upper border for path length
                    # (to detect oscillation)
                    if path_length > 2*len(rollout_data.expert_paths[map_id][path_id]):
                        del open_paths[current_open_task_id]
                        continue
                    
                    # ensure that perceptive area lies completely within grid world
                    if pos[0] >= 3*rollout_data.grids[map_id].size()[0]//4 or pos[0] < rollout_data.grids[map_id].size()[0]//4 or pos[1] >= 3*rollout_data.grids[map_id].size()[1]//4 or pos[1] < rollout_data.grids[map_id].size()[1]//4:
                        del open_paths[current_open_task_id]
                        continue
                    
                    # check collision
                    if rollout_data.grids[map_id][pos[0],pos[1]] == 1:
                        del open_paths[current_open_task_id]
                        continue
                    
                    current_open_task_id += 1
            
            if not validation:
                if path_length % 20 == 0:
                    print("Computed paths up to length ", path_length)
                
        pool.close()
        
        # count successful paths
        num_successful = 0
        for i in range(rollout_data.num_examples):
            for j in range(rollout_data.num_paths_per_map):
                paths[i][j] = torch.stack(paths[i][j], dim=0)
                if success[i][j]:
                    num_successful += 1
                    if not validation:
                        # compare length of network and expert paths
                        diff += get_path_length(paths[i][j], dim=2)-get_path_length(rollout_data.expert_paths[i][j],dim=2)
                        net_length += get_path_length(paths[i][j], dim=2)
                        expert_length += get_path_length(rollout_data.expert_paths[i][j], dim=2)

        if not validation:
            print("Success: ", num_successful/ len(rollout_data))
            print("Path length (network): ", net_length)
            print("Path length (expert): ", expert_length)
            print("Average absolute path difference: ", diff/ num_successful)
            print("average relative path difference: ", net_length/expert_length)
            print("Duration: ", time.time() - start_time)
            print("")
        return num_successful/ len(rollout_data)

def _get_inputs(parameters):
        with torch.no_grad():
            map_index, pos_t1, pos_t2 = parameters
            rollout_img = rollout_data.grids[map_index]
            
            # environment map
            occ_map = rollout_img[pos_t1[0] - rollout_data.size//2:pos_t1[0]+rollout_data.size//2,pos_t1[1] - rollout_data.size//2:pos_t1[1]+rollout_data.size//2]
            
            # goal map
            goal_map = torch.FloatTensor(rollout_data.size, rollout_data.size).fill_(0)
            local_pos_t2 = pos_t2[0:2] - pos_t1[0:2] + rollout_data.centre
            if local_pos_t2[0].item() >= 0 and local_pos_t2[0].item() < rollout_data.size and local_pos_t2[1].item() >= 0 and local_pos_t2[1].item() < rollout_data.size:
                goal_map[local_pos_t2[0], local_pos_t2[1]] = 1
            
            return occ_map.unsqueeze(0), goal_map.unsqueeze(0)
    
def batch(iterable, step=1):
    length = len(iterable)
    for i in range(0, length, step):
        yield iterable[i:min(i + step, length)]

