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
        '--iterations',
        type=int,
        help='For 2D: Number of training epochs. For 3D: Number of learning rate cycles.')
    parser.add_argument(
        '--batch',
        type=int,
        default=128,
        help='Batch size')
    parser.add_argument(
        '--lr',
        type=int,
        default=0.001,
        help='Learning rate')
    parser.add_argument(
        '--k',
        type=int,
        help='Number of iterations for the value iteration algorithm.')
    parser.add_argument(
        '--validation_step',
        type=int,
        default=20,
        help='Number of epochs between two tests on the validation set (Only for 2D).')
    parser.add_argument(
        '--lr_cycle_length',
        type=int,
        default=48,
        help='Length of first learning rate cylce in epochs (only for 3D).')
    parser.add_argument(
        '--lr_cycle_decay',
        type=int,
        default=0.95,
        help='Factor to decay initial learning rate after each learning rate cycles (only for 3D).')
    parser.add_argument(
        '--lr_cycle_increase',
        type=int,
        default=1.5,
        help=' Factor which increases the length of a learning rate cycle after each learning rate cycle (only for 3D).')
    parser.add_argument(
        '--print_stat',
        type=bool,
        default=True,
        help='Print training statistics to file. [True, False]')
    parser.add_argument(
        '--plot_stat',
        type=bool,
        default=True,
        help='Plot training statistics. [True, False]')
    param = parser.parse_args()
    
    
    if param.iterations is None:
        if param.dim==2:
            param.iterations=1500
        else:
            pram.iterations=7
    
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
        net.train(num_iterations=param.iterations, batch_size=param.batch, validation_step=param.validation_step, lr=param.lr,plot_curves=param.plot_stat, print_stat=param.print_stat)
    else:
        net = Abstraction_VIN_3D(param.size, k = param.k)
        
        # load network state
        if os.path.isfile('network/%s.pt' % net.name):
            print("Using existing network.")
            net.load_state_dict(torch.load('network/%s.pt' % net.name))
        else:
            print("New network initialized.")
            
        net.to(device)
        net.train(num_iterations=param.iterations, batch_size=param.batch, lr=param.lr, lr_cycle_length=param.lr_cycle_length, T_mult=param.lr_cycle_increase, lr_decay_factor=param.lr_cycle_decay, plot_curves=param.plot_stat, print_stat=param.print_stat)
