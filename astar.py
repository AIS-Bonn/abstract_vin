import numpy as np
import heapq as heap
import torch
from math import cos, sin

# returns a list of valid neighbor poses
# Input:    -example: environment (of type LabeledExample)
#           -current: current pose
#           -closed: information about already closed poses (one bool for each pose)
#           -dim: 2 for 2D grid world tasks or 3 for 3D locomotion planning task
# Output: list of valid neighbor poses, each pose is a tuple containing the discretized x and y coordinates (and orientation)
def _get_valid_neighbours(example, current, closed, dim=3):
    if dim == 3:
        # parameters for sub-sampling rotations and diagonal movements for collision detection
        subsamples_move=4
        subsample_step_size_move = 1./subsamples_move
        subsamples_rotation=4
        subsample_step_size_rotation = 1./subsamples_rotation
    
    neighbours = []
    
    # neighbors through axis aligned moves
    for x,y in [[-1,0],[0,-1],[0,1],[1,0]]:
        if dim == 2:
            neighbour = (current[0] + x,current[1]+y)
            
            # skip closed poses
            if closed[neighbour[0],neighbour[1]]==1:
                continue
            
            # do not move out of robot's receptive field
            if neighbour[0] >= 3*example.size//4 or neighbour[1] >= 3*example.size//4:
                continue
            
            if neighbour[0] < example.size//4 or neighbour[1] < example.size//4:
                continue
            
            # check collision
            if example.map[neighbour[0],neighbour[1]] == 1:
                continue
            
        elif dim == 3:
            neighbour = (current[0] + x,current[1]+y, current[2])
            
            # skip closed poses
            if closed[neighbour[0],neighbour[1],neighbour[2]]==1:
                continue
        
            # get wheel positions
            fl,fr,bl,br = get_wheel_positions(neighbour, example.rotation_step_size, example.leg_x, example.leg_y)
            
            # get index of corresponding grid cell
            fl,fr,bl,br = fl.round().long(),fr.round().long(),bl.round().long(),br.round().long()
        
            # do not move out of robot's receptive field
            if fl[0] >= 3*example.size//4 or fl[1] >= 3*example.size//4 or fr[0] >= 3*example.size//4 or fr[1] >= 3*example.size//4 or bl[0] >= 3*example.size//4 or bl[1] >= 3*example.size//4 or br[0] >= 3*example.size//4 or br[1] >= 3*example.size//4:
                continue

            if fl[0] < example.size//4 or fl[1] < example.size//4 or fr[0] < example.size//4 or fr[1] < example.size//4 or bl[0] < example.size//4 or bl[1] < example.size//4 or br[0] < example.size//4 or br[1] < example.size//4:
                continue
            
            # check collision for each wheel
            if example.map[fl[0],fl[1]] == 1 or example.map[fr[0],fr[1]] == 1 or example.map[bl[0],bl[1]] == 1 or example.map[br[0],br[1]] == 1:
                continue
        
        neighbours.append(neighbour)
        
    # neighbors through diagonal moves
    for x,y in [[-1,-1],[-1,1],[1,-1],[1,1]]:
        if dim == 2:
            neighbour = (current[0] + x,current[1]+y)
            
            # skip closed poses
            if closed[neighbour[0],neighbour[1]]==1:
                continue
            
            # do not move out of robot's receptive field
            if neighbour[0] >= 3*example.size//4 or neighbour[1] >= 3*example.size//4:
                continue

            if neighbour[0] < example.size//4 or neighbour[1] < example.size//4:
                continue
            
            # check collision
            if example.map[neighbour[0],neighbour[1]] == 1:
                continue
            
            neighbours.append(neighbour)
            
        elif dim == 3:
            neighbour = (current[0] + x,current[1]+y, current[2])
            
            if closed[neighbour[0],neighbour[1],neighbour[2]]==1:
                continue
            
            # get wheel positions
            fl,fr,bl,br = get_wheel_positions(neighbour, example.rotation_step_size, example.leg_x, example.leg_y)
            fl,fr,bl,br = fl.round().long(),fr.round().long(),bl.round().long(),br.round().long()
            
            # do not move out of robot's receptive field
            if fl[0] >= 3*example.size//4 or fl[1] >= 3*example.size//4 or fr[0] >= 3*example.size//4 or fr[1] >= 3*example.size//4 or bl[0] >= 3*example.size//4 or bl[1] >= 3*example.size//4 or br[0] >= 3*example.size//4 or br[1] >= 3*example.size//4:
                continue

            if fl[0] < example.size//4 or fl[1] < example.size//4 or fr[0] < example.size//4 or fr[1] < example.size//4 or bl[0] < example.size//4 or bl[1] < example.size//4 or br[0] < example.size//4 or br[1] < example.size//4:
                continue
            
            # check collisions for each wheel
            if example.map[fl[0],fl[1]] == 1 or example.map[fr[0],fr[1]] == 1 or example.map[bl[0],bl[1]] == 1 or example.map[br[0],br[1]] == 1:
                continue
            
            # subsample movement to detect collisions for intermediate poses
            valid = True
            for step in range(1,subsamples_move):
                tmp = (current[0] + step*subsample_step_size_move*x,current[1]+step*subsample_step_size_move*y, current[2])

                # get wheel positions
                fl_tmp,fr_tmp,bl_tmp,br_tmp = get_wheel_positions(tmp, example.rotation_step_size, example.leg_x, example.leg_y)
                fl_tmp,fr_tmp,bl_tmp,br_tmp = fl_tmp.round().long(),fr_tmp.round().long(),bl_tmp.round().long(),br_tmp.round().long()

                # check collisions for each wheel
                if example.map[fl_tmp[0],fl_tmp[1]] == 1 or example.map[fr_tmp[0],fr_tmp[1]] == 1 or example.map[bl_tmp[0],bl_tmp[1]] == 1 or example.map[br_tmp[0],br_tmp[1]] == 1:
                    valid = False
                    break
                
            if valid:
                neighbours.append(neighbour)
    
    if dim == 3:
        # neighbors through rotations
        for turn in [-1,1]:
            new_orientation = current[2] + turn
            if new_orientation < 0:
                new_orientation += example.num_orientations
            elif new_orientation >= example.num_orientations:
                new_orientation -= example.num_orientations
                
            neighbour = (current[0],current[1], new_orientation) 
            
            # skip closed poses
            if closed[neighbour[0],neighbour[1],neighbour[2]]==1:
                continue
            
            # get wheel positions
            fl,fr,bl,br = get_wheel_positions(neighbour, example.rotation_step_size, example.leg_x, example.leg_y)
            fl,fr,bl,br = fl.round().long(),fr.round().long(),bl.round().long(),br.round().long()
            
            # do not move out of robot's receptive field
            if fl[0] >= 3*example.size//4 or fl[1] >= 3*example.size//4 or fr[0] >= 3*example.size//4 or fr[1] >= 3*example.size//4 or bl[0] >= 3*example.size//4 or bl[1] >= 3*example.size//4 or br[0] >= 3*example.size//4 or br[1] >= 3*example.size//4:
                continue

            if fl[0] < example.size//4 or fl[1] < example.size//4 or fr[0] < example.size//4 or fr[1] < example.size//4 or bl[0] < example.size//4 or bl[1] < example.size//4 or br[0] < example.size//4 or br[1] < example.size//4:
                continue
            
            # check collisions for each wheel
            if example.map[fl[0],fl[1]] == 1 or example.map[fr[0],fr[1]] == 1 or example.map[bl[0],bl[1]] == 1 or example.map[br[0],br[1]] == 1:
                continue
            
            # subsample movement to detect collisions for intermediate poses
            valid = True
            for step in range(1,subsamples_rotation):
                tmp_orientation = current[2] + step*subsample_step_size_rotation*turn
                if tmp_orientation < 0:
                    tmp_orientation += example.num_orientations
                elif tmp_orientation >= example.num_orientations:
                    tmp_orientation -= example.num_orientations
                tmp = (current[0],current[1], tmp_orientation)
                
                # get wheel positions
                fl_tmp,fr_tmp,bl_tmp,br_tmp = get_wheel_positions(tmp, example.rotation_step_size, example.leg_x, example.leg_y)
                fl_tmp,fr_tmp,bl_tmp,br_tmp = fl_tmp.round().long(),fr_tmp.round().long(),bl_tmp.round().long(),br_tmp.round().long()
                
                # do not move out of robot's receptive field
                if fl_tmp[0] >= 3*example.size//4 or fl_tmp[1] >= 3*example.size//4 or fr_tmp[0] >= 3*example.size//4 or fr_tmp[1] >= 3*example.size//4 or bl_tmp[0] >= 3*example.size//4 or bl[1] >= 3*example.size//4 or br_tmp[0] >= 3*example.size//4 or br_tmp[1] >= 3*example.size//4:
                    continue

                if fl_tmp[0] < example.size//4 or fl_tmp[1] < example.size//4 or fr_tmp[0] < example.size//4 or fr_tmp[1] < example.size//4 or bl_tmp[0] < example.size//4 or bl_tmp[1] < example.size//4 or br_tmp[0] < example.size//4 or br_tmp[1] < example.size//4:
                    continue
                
                # check collisions for each wheel
                if example.map[fl_tmp[0],fl_tmp[1]] == 1 or example.map[fr_tmp[0],fr_tmp[1]] == 1 or example.map[bl_tmp[0],bl_tmp[1]] == 1 or example.map[br_tmp[0],br_tmp[1]] == 1:
                    valid = False
                    break
                
            if valid:
                neighbours.append(neighbour)
        
    return neighbours

# A* planner for the 2D grid world task
# Input:    -example: environment (of type LabeledExample)
# Output: optimal path (list of grid cell coordinates)
def astar_planner_2d(example):    
    path = []
    
    start = (example.start[0],example.start[1])
    if example.goal == ():
        print("No goal set.")
        path.append(start)
        return path
    goal = (example.goal[0],example.goal[1])
    g_values = np.full((example.size, example.size), float('Inf'))
    g_values[start[0],start[1]]=0

    predecessor_x = np.empty([example.size, example.size], dtype=int)
    predecessor_y = np.empty([example.size, example.size], dtype=int)
    
    closed = np.zeros((example.size, example.size))
        
    #push start to open_list
    open_list = []
    heap.heappush(open_list, (0,start))
        
    while open_list:
        current = heap.heappop(open_list)[1]
        
        # check if current pose is invalid because key was decreased
        # (as there is no decrease key operation, a duplicated pose with new (smaller) key is inserted)
        if closed[current[0],current[1]] == 1:
            continue
        
        if current[0] == goal[0] and current[1] == goal[1]:
            #while current in predecessor:
            while (current[0] != start[0] or current[1] != start[1]):
                path.append(torch.tensor([current[0],current[1]]))
                current = (predecessor_x[current[0],current[1]], predecessor_y[current[0],current[1]])
            
            path.append(torch.tensor([start[0],start[1]]))
            return path
        
        closed[current[0],current[1]] = 1

        #expand current node
        neighbours = _get_valid_neighbours(example, current, closed, dim=2)
        for neighbour in neighbours:              
            g = g_values[current[0],current[1]] + np.sqrt((current[0]-neighbour[0])**2 + (current[1]-neighbour[1])**2)
            
            if g >= g_values[neighbour[0],neighbour[1]]:
                continue
            else:
                predecessor_x[neighbour[0],neighbour[1]] = current[0]
                predecessor_y[neighbour[0],neighbour[1]] = current[1]

            g_values[neighbour[0],neighbour[1]]=g
            f = g + np.sqrt((goal[0]-neighbour[0])**2+(goal[1]-neighbour[1])**2)
            heap.heappush(open_list, (f,neighbour))
            
    path.append(torch.tensor([start[0],start[1]]))

    return path

# Dijkstra planner for the 2D grid world task
# Input:    -example: environment (of type LabeledExample)
# Output:   -costs to reach each state
#           -array containing predecessor for each state
def dijkstra_planner_2d(example):    
    start = (example.start[0],example.start[1])
    g_values = np.full((example.size, example.size), float('Inf'))
    g_values[start[0],start[1]]=0

    predecessor = np.empty([2,example.size, example.size], dtype=int)
    
    closed = np.zeros((example.size, example.size))
        
    #push start to open_list
    open_list = []
    heap.heappush(open_list, (0,start))
        
    while open_list:
        current = heap.heappop(open_list)[1]
        
        # check if current pose is invalid because key was decreased
        # (as there is no decrease key operation, a duplicated pose with new (smaller) key is inserted)
        if closed[current[0],current[1]] == 1:
            continue
        
        closed[current[0],current[1]] = 1
        
        neighbours = _get_valid_neighbours(example, current, closed, dim=2)
        for neighbour in neighbours:
            g = g_values[current[0],current[1]].item() + np.sqrt((current[0]-neighbour[0])**2 + (current[1]-neighbour[1])**2)
            
            if g >= g_values[neighbour[0],neighbour[1]]:
                continue
            else:
                predecessor[0,neighbour[0],neighbour[1]] = current[0]
                predecessor[1,neighbour[0],neighbour[1]] = current[1]
            
            g_values[neighbour[0],neighbour[1]]=g
            heap.heappush(open_list, (g,neighbour))
            
    return g_values, predecessor

# A* planner for the 3D locomotion planning task
# Input:    -example: environment (of type LabeledExample)
# Output: optimal path (list of grid cell coordinates)
def astar_planner_3d(example):    
    path = []
    
    start = (example.start[0],example.start[1], example.orientation)
    if example.goal == ():
        print("No goal set.")
        path.append(start)
        return path
    goal = (example.goal[0],example.goal[1], example.goal_orientation)
    g_values = np.full((example.size, example.size, example.num_orientations), float('Inf'))
    g_values[start[0],start[1],start[2]]=0
    predecessor_x = np.empty([example.size, example.size, example.num_orientations], dtype=int)
    predecessor_y = np.empty([example.size, example.size, example.num_orientations], dtype=int)
    predecessor_theta = np.empty([example.size, example.size, example.num_orientations], dtype=int)
    
    closed = np.zeros((example.size, example.size, example.num_orientations))
        
    #push start to open_list
    open_list = []
    heap.heappush(open_list, (0,start))
        
    while open_list:
        current = heap.heappop(open_list)[1]
        # check if current pose is invalid because key was decreased
        # (as there is no decrease key operation, a duplicated pose with new (smaller) key is inserted)
        if closed[current[0],current[1],current[2]] == 1:
            continue
        
        #print(current)
        if current[0] == goal[0] and current[1] == goal[1] and current[2] == goal[2]:
            #while current in predecessor:
            while (current[0] != start[0] or current[1] != start[1] or current[2] != start[2]):
                path.append(torch.tensor([current[0],current[1],current[2]]))
                #current = predecessor[current]
                #print(current)
                current = (predecessor_x[current[0],current[1],current[2]], predecessor_y[current[0],current[1],current[2]],predecessor_theta[current[0],current[1], current[2]])
            
            path.append(torch.tensor([start[0],start[1],start[2]]))
            #print("Path found")
            return path
        
        closed[current[0],current[1],current[2]] = 1

        #expand current node
        neighbours = _get_valid_neighbours(example, current, closed)
        for neighbour in neighbours:              
            g = g_values[current[0],current[1],current[2]] + np.sqrt((current[0]-neighbour[0])**2 + (current[1]-neighbour[1])**2)
            if current[2] != neighbour[2]:
                g += np.sqrt(example.leg_x**2+example.leg_y**2)*example.rotation_step_size
            
            if g >= g_values[neighbour[0],neighbour[1],neighbour[2]]:
                continue
            else:
                predecessor_x[neighbour[0],neighbour[1],neighbour[2]] = current[0]
                predecessor_y[neighbour[0],neighbour[1],neighbour[2]] = current[1]
                predecessor_theta[neighbour[0],neighbour[1],neighbour[2]] = current[2]

            g_values[neighbour[0],neighbour[1],neighbour[2]]=g
            f = g + np.sqrt((goal[0]-neighbour[0])**2+(goal[1]-neighbour[1])**2) 
            if(goal[2] != neighbour[2]):
                orientation_diff = np.abs(goal[2]-neighbour[2])
                if orientation_diff > example.num_orientations/2:
                    orientation_diff = example.num_orientations-orientation_diff
                f += np.sqrt(example.leg_x**2+example.leg_y**2)*orientation_diff*example.rotation_step_size
            heap.heappush(open_list, (f,neighbour))
            
    path.append(torch.tensor([start[0],start[1],start[2]]))
    return path

# Dijkstra planner for the 3D locomotion planning task
# Input:    -example: environment (of type LabeledExample)
# Output:   -costs to reach each state
#           -array containing predecessor (on optimal path) for each state
def dijkstra_planner_3d(example):    
    start = (example.start[0],example.start[1], example.orientation)
    g_values = np.full((example.size, example.size, example.num_orientations), float('Inf'))
    g_values[start[0],start[1],start[2]]=0
    predecessor = np.empty([3,example.size, example.size, example.num_orientations], dtype=int)
    
    closed = np.zeros((example.size, example.size, example.num_orientations))
        
    #push start to open_list
    open_list = []
    heap.heappush(open_list, (0,start))
        
    while open_list:
        current = heap.heappop(open_list)[1]
        
        # check if current pose is invalid because key was decreased
        ## (as there is no decrease key operation, a duplicated pose with new (smaller) key is inserted)
        if closed[current[0],current[1],current[2]] == 1:
            continue
        
        closed[current[0],current[1],current[2]] = 1
        
        neighbours = _get_valid_neighbours(example, current, closed)
        for neighbour in neighbours:
            g = g_values[current[0],current[1],current[2]].item() + np.sqrt((current[0]-neighbour[0])**2 + (current[1]-neighbour[1])**2)
            if current[2] != neighbour[2]:
                g += np.sqrt(example.leg_x**2+example.leg_y**2)*example.rotation_step_size
            
            if g >= g_values[neighbour[0],neighbour[1],neighbour[2]]:
                continue
            else:
                predecessor[0,neighbour[0],neighbour[1],neighbour[2]] = current[0]
                predecessor[1,neighbour[0],neighbour[1],neighbour[2]] = current[1]
                predecessor[2,neighbour[0],neighbour[1],neighbour[2]] = current[2]
            
            g_values[neighbour[0],neighbour[1],neighbour[2]]=g
            heap.heappush(open_list, (g,neighbour))
            
    return g_values, predecessor

# returns the current wheel positions (for the 3D task)
# Input:        -pose: robot base position and orientation
#               -rotation_step_size: difference between two adjacent discretized orientations
#               -leg_x, leg_y: absolute values of distance between wheel and robot base position
# Output: wheel positions (float values)
def get_wheel_positions(pose,rotation_step_size, leg_x, leg_y):
    base = torch.tensor([pose[0],pose[1]]).float()
    theta = pose[2]*rotation_step_size

    # local wheel coordinates
    local_fl = torch.tensor([leg_x,leg_y]).float()
    local_fr = torch.tensor([leg_x,-leg_y]).float()
    local_bl = torch.tensor([-leg_x,leg_y]).float()
    local_br = torch.tensor([-leg_x,-leg_y]).float()
    
    # global coordinates
    rotation = torch.tensor([[cos(-theta), sin(-theta)],[-sin(-theta), cos(-theta)]])
    fl = base + torch.mv(rotation,local_fl)
    fr = base + torch.mv(rotation,local_fr)
    bl = base + torch.mv(rotation,local_bl)
    br = base + torch.mv(rotation,local_br)

    return (fl,fr,bl,br)
