import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
from copy import deepcopy
from maze_utils import Maze


# method for generating history of trajectory
def generate_trajectory_history(data, history_length = 0):
    # data: position data
    # history_length: length of history to include in trajectory
    
    data = data.unsqueeze(-1)
    
    history_data = []
    # iterate over the position data
    for i in range(history_length, data.size(dim = 0)):
        # generate trajectory history and append to list
        new = data[i-history_length:i,:,:].transpose(0,-1).transpose(1,-1)
        history_data.append(new)
    
    # concatenate data points of trajectory with history and return
    return torch.cat(history_data, dim = 0).float()


# xvar = 0.2857, yvar = 0.6330
# method for generating out maze data using random walk
def generate_out_maze_data(xmin, xmax, ymin, ymax, n_points, plot = False):
    # xmin: lower bound on x-axis
    # xmax: upper bound on x-axis
    # ymin: lower bound on y-axis
    # ymax: upper bound on y-axis
    
    # create intial point for random walk
    init = torch.zeros((1,2))
    init[:,0].uniform_(xmin, xmax)
    init[:,1].uniform_(ymin, ymax)
    
    # create covariance matrix
    covar = torch.zeros((2,2))
    covar[0,0] = 0.2857
    covar[1,1] = 0.6330
    
    curr = init
    data = []
    data.append(curr)
    
    stime = time.time()
    
    for i in range(1,n_points):
        # update prev
        prev = curr
        
        # logic for ensuring next position is within bounds
        done = False
        while not done:
            # generate next position
            curr = MultivariateNormal(prev, covar).sample()
            # conditions for ensuring next position is within bounds
            if curr[:,0] > xmin and curr[:,0] < xmax and curr[:,1] > ymin and curr[:,1] < ymax:
                # break out of while loop
                done = True
            
        # save valid next position to list
        data.append(curr)
        
        # printing
        if i % 100000 == 0:
            print('time elapsed: %.1f' % (time.time() - stime))
    
    # concatenate random walk data
    data = torch.cat(data, dim = 0)
    
    # plotting
    if plot:
        plt.figure()
        plt.plot(data[:,0], data[:,1], '0.4')
        
    
    return data


# method for generating in-maze data by adding small Gaussian noise
def generate_in_maze_data(in_maze_data, variance, history_length, iterations = 1):
    # in_maze_data: existing in-maze data
    # variance: variance of noise to add to in_maze_data
    # history_length: desired history length of output data
    
    # create covariance matrix
    covar = torch.zeros((2,2))
    covar[0,0], covar[1,1] = variance, variance
    
    # initialize new data
    new_data = deepcopy(in_maze_data)
    
    if iterations == 1:
        # add small Gaussian noise
        noisy_data = MultivariateNormal(new_data, covar).sample()
        # generate trajectory history
        noisy_history = generate_trajectory_history(noisy_data, history_length = history_length)
    else:
        noisy_history = []
        for i in range(iterations):
            # add small Gaussian noise
            noisy_data = MultivariateNormal(new_data, covar).sample()
            # generate trajectory history and append to list
            noisy_history.append(
                generate_trajectory_history(noisy_data, history_length = history_length))
        # concatenate data
        noisy_history = torch.cat(noisy_history, dim = 0)
        
    return noisy_history



if __name__ == '__main__':
    
    # boolean toggle indicating whether to generate in-maze data or out-maze data
    gen_in_maze_data = True
    
    
    if gen_in_maze_data:
    
        # load Wmaze sessions    
    
        wm1 = Maze(
            name = 'Remy',
            session = (36,1),
            n_points = 1500000,
            Fs = 3000,
            rem_insig_chans = True,
            threshold = 0,
            )
        
        wm2 = Maze(
            name = 'Remy', 
            session = (36,3), 
            n_points = 1500000,
            Fs = 3000,
            rem_insig_chans = True,
            threshold = 0,
            )
        
        wm3 = Maze(
            name = 'Remy', 
            session = (37,1), 
            n_points = 1500000,
            Fs = 3000,
            rem_insig_chans = True,
            threshold = 0,
            )
    
        # get position data from each session with history length 1
        dat1 = wm1.generate_position_history(history_length = 1)[:,:,:2].squeeze()
        dat2 = wm2.generate_position_history(history_length = 1)[:,:,:2].squeeze()
        dat3 = wm3.generate_position_history(history_length = 1)[:,:,:2].squeeze()
        
        # desired history length
        hl = 64
        
        
        # add varying levels of noise to provided in-maze data
        
        new11 = generate_in_maze_data(dat1, variance = 1e-3, history_length = hl, iterations = 2)
        new12 = generate_in_maze_data(dat1, variance = 1e-4, history_length = hl, iterations = 2)
        new13 = generate_in_maze_data(dat1, variance = 1e-5, history_length = hl, iterations = 3)
        new14 = generate_in_maze_data(dat1, variance = 1e-6, history_length = hl, iterations = 3)
        
        new21 = generate_in_maze_data(dat2, variance = 1e-3, history_length = hl, iterations = 2)
        new22 = generate_in_maze_data(dat2, variance = 1e-4, history_length = hl, iterations = 2)
        new23 = generate_in_maze_data(dat2, variance = 1e-5, history_length = hl, iterations = 3)
        new24 = generate_in_maze_data(dat2, variance = 1e-6, history_length = hl, iterations = 3)
        
        new31 = generate_in_maze_data(dat3, variance = 1e-3, history_length = hl, iterations = 2)
        new32 = generate_in_maze_data(dat3, variance = 1e-4, history_length = hl, iterations = 2)
        new33 = generate_in_maze_data(dat3, variance = 1e-5, history_length = hl, iterations = 3)
        new34 = generate_in_maze_data(dat3, variance = 1e-6, history_length = hl, iterations = 3)
    
        # get position data from each session with desired history length
        dat1 = wm1.generate_position_history(history_length = hl)[:,:,:2]
        dat2 = wm2.generate_position_history(history_length = hl)[:,:,:2]
        dat3 = wm3.generate_position_history(history_length = hl)[:,:,:2]
        
        # concatenate all in-maze data
        in_data = torch.cat([
            dat1, new11, new12, new13, new14,
            dat2, new21, new22, new23, new24,
            dat3, new31, new32, new33, new34,
            ], dim = 0)
        
        print(in_data.size())
        
        # save generated in-maze data
        #torch.save(in_data, f'Datasets/in_data_HL{hl}.pt')

    else:
        # generate out-maze data
        data = generate_out_maze_data(
            xmin = -50, xmax = 150, ymin = -50, ymax = 150, 
            n_points = 500000, plot = True,
            )
        
        print('\n', data, data.size())
        
        # save out-maze data
        #torch.save(data, 'InMaze/out_trajectory.pt')

















