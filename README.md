# Value Iteration Networks on Multiple Levels of Abstraction

Value Iteration Networks (VINs) on Multiple Levels of Abstraction combine VINs by Tamar et. al (NIPS 2016) with the idea of planning on multiple levels of abstraction to obtain a learning-based planner which is capable of solving planning tasks on larger state spaces.

This repository contains PyTorch implementations applying the method to 2D grid world planning for a point-like agent and 3D locomotion planning with footprint consideration for a robot that can perform omnidirectional driving.
In detail, it contains implementations of
- Value Iteration Networks for 2D grid world planning,
- Hierarchical Value Iteration Networks for 2D grid world planning,
- Value Iteration Networks on multiple levels of abstraction for 2D grid world planning, and
- Value Iteration Networks on multiple levels of abstraction for 3D locomotion planning.


Please see the see the respective paper for further information:
"Value Iteration Networks on Multiple Levels of Abstraction" by Daniel Schleich, Tobias Klamt, and Sven Behnke.


## Installation
This repository requires following packages:
- [Python](https://www.python.org/) = 2.7
- [PyTorch](http://pytorch.org/) >= 0.4.1
- [Numpy](https://pypi.python.org/pypi/numpy) >= 1.14.3
- [Matplotlib](https://matplotlib.org/users/installing.html) >= 2.2.2


## Generate Datasets
Datasets for training and evaluation can be generated using the ```generate_dataset.py``` script, e.g
```
python generate_dataset.py --dim 2 --size 32
```
**Flags**: 
- `dim`: Dimensionality of the planning task: '2' for 2D grid worlds and '3' for 3D locomotion planning
- `size`: The size of input maps. [32, 64]
- `type`: Type of dataset. One of [training, validation, evaluation, all]. Default: 'all'
- `num_grids`: Number of different grid worlds. If `type` is set to all, default values are used: 5000 for training and 715 each for validation and evaluation.
- `paths_per_grid`: Number of different paths for each grid world. Default: 7.
- `num_workers`: Number of workers for parallel grid generation. Recommended: Set to number of CPU cores.

## Train the Networks  (2D grid world)
The networks can be trained using the `train.py` script:
#### VIN
```
python train.py --dim 2 --net VIN --size 32 --iterations 100 --lr 0.001 --batch 128
```

#### Hierarchical VIN  (2D grid world)
```
python train.py --dim 2 --net HVIN --size 32 --iterations 100 --lr 0.001 --batch 128
```

#### VIN on multiple abstraction levels (2D grid world)
```
python train.py --dim 2 --net Abstraction_VIN --size 32 --iterations 100 --lr 0.001 --batch 128
```

#### VIN on multiple abstraction levels (3D locomotion planning)
```
python train.py --dim 3 --net Abstraction_VIN --size 32 --iterations 7 --lr 0.001 --batch 128
```

**Flags**: 
- `dim`: Dimensionality of the planning task: '2' for 2D grid worlds and '3' for 3D locomotion planning
- `size`: The size of input maps. [32, 64]
- `net`: Network to be trained: For 2D grid worlds one of [VIN, HVIN, Abstraction_VIN]. For 3D only 'Abstraction_VIN'.
- `batch`: Batch size. Default: 128.
- `lr`: Learning rate. Default: 0.001.
- `validation_step`: Number of epochs between two tests on the validation set (only for 2D).
- `lr_cycle_length`:  Length of first learning rate cylce in epochs (only for 3D).
- `lr_cycle_decay`:  Factor to decay initial learning rate after each learning rate cycle (only for 3D).
- `lr_cycle_increase`:  Factor which increases the length of a learning rate cycle after each cycle (only for 3D).
- `k`: Number of value iterations.
- `print_stat`: Print training statistics to file (in folder learning_curves). One of [True, False].
- `plot_stat`: Plot training statistics to file (in folder learning_curves). One of [True, False].


## Test the Networks (requires training first)
The networks can be tested using the `test.py` script:
#### VIN
```
python test.py --dim 2 --net VIN --size 32 --batch 128
```

#### Hierarchical VIN  (2D grid world)
```
python test.py --dim 2 --net HVIN --size 32 --batch 128
```

#### VIN on multiple abstraction levels (2D grid world)
```
python test.py --dim 2 --net Abstraction_VIN --size 32 --batch 128
```

#### VIN on multiple abstraction levels (3D locomotion planning)
```
python test.py --dim 3 --net Abstraction_VIN --size 32 --batch 128

**Flags**: 
- `dim`: Dimensionality of the planning task: '2' for 2D grid worlds and '3' for 3D locomotion planning
- `size`: The size of input maps. [32, 64]
- `net`: Network to be trained: For 2D grid worlds one of [VIN, HVIN, Abstraction_VIN]. For 3D only 'Abstraction_VIN'.
- `batch`: Batch size. Default: 128.
- `k`: Number of value iterations.
- `workers`: Number of workers for parallel evaluation of multiple tasks.


## Visualize Paths (requires training first)
The resulting paths can be plotted using the `visualize.py` script:

#### 2D Grid World
```
python visualize.py --dim 2 --size 32 --num 5
```

#### 3D Locomotion Planning
```
python visualize.py --dim 3 --size 32 --num 5

**Flags**: 
- `dim`: Dimensionality of the planning task: '2' for 2D grid worlds and '3' for 3D locomotion planning
- `size`: The size of input maps. [32, 64]
- `num`: Number of different paths that shall be plotted. (Only if no explicit map and path ids are given)
- `map_id`: Index of the map that shall be plotted. If none is given, it will be chosen randomly.
- `path_id`: Index of the path that shall be plotted. If none is given, it will be chosen randomly.


## Acknowledgement
Many thanks to Kent Sommer, whose nice PyTorch implementation of Value Iteration Networks served as a starting point for our own implementations:
https://github.com/kentsommer/pytorch-value-iteration-networks


## License
This software is released under BSD-3.


## Contact
If you have any questions, mail Daniel Schleich (schleich@uni-bonn.de).
