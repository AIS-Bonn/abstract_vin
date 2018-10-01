import torch
import argparse
import os.path

from vin.nets_2d import VIN, HVIN, Abstraction_VIN_2D
from vin.net_3d import Abstraction_VIN_3D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=2,
        help='Dimensionality of the grid: either 2 for the 2D grid world task or 3 for 3D locomotion planning.')
    parser.add_argument('--size', type=int, default=32, help='Size of grid world [32, 64]')
    parser.add_argument(
        '--net',
        type=str,
        default='Abstraction_VIN',
        help='Network to be trained: VIN (only for 2D), HVIN (only for 2D) or Abstraction_VIN (for 2D and 3D)')
    parser.add_argument(
        '--batch',
        type=int,
        default=128,
        help='Batch size')
    parser.add_argument(
        '--k',
        type=int,
        help='Number of iterations for the value iteration algorithm.')
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers for unrolling multiple paths in parallel.')
    param = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    # train net
    if param.dim == 2:
        if param.net == 'VIN':
            net = VIN(param.size, k = param.k)
        elif param.net == 'HVIN':
            net = HVIN(param.size, k = param.k)
        else:
            net = Abstraction_VIN_2D(param.size, k = param.k)
        
        # load network state
        if os.path.isfile('network/%s.pt' % net.name):
            print("Using existing network.")
            net.load_state_dict(torch.load('network/%s.pt' % net.name))
        else:
            print("New network initialized.")
               
        net.to(device)
        net.test(batch_size=param.batch, validation=False)
        net.rollout(batch_size = param.batch, validation=False, num_workers=param.workers)
    else:
        net = Abstraction_VIN_3D(param.size, k = param.k)
        
        # load network state
        if os.path.isfile('network/%s.pt' % net.name):
            print("Using existing network.")
            net.load_state_dict(torch.load('network/%s.pt' % net.name))
        else:
            print("New network initialized.")
            
        net.to(device)
        net.test(batch_size=param.batch, validation=False)
        net.rollout(batch_size = param.batch, validation=False, num_workers=param.workers)
