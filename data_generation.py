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



def data_to_index(data, xmin, ymin, resolution, unique = True):
    
    idxs = torch.zeros_like(data)
    idxs[:,0] = ((data[:,0] - xmin) / resolution).floor()
    idxs[:,1] = ((data[:,1] - ymin) / resolution).floor()
    
    if unique:
        return idxs.unique(dim = 0).long()
    else:
        return idxs.long()



def index_to_data(indices, xmin, ymin, resolution):
    
    data = torch.zeros_like(indices)
    data[:,0] = resolution*(indices[:,0]+0.5) + xmin
    data[:,1] = resolution*(indices[:,1]+0.5) + ymin
        
    return data



def generate_out_classifier_data(
        maze_data, xmin, xmax, ymin, ymax, 
        resolution = 0.1, n_samples = 0, plot = False):
    
    in_ix = data_to_index(maze_data, xmin, ymin, resolution = resolution)
    in_ix_min = in_ix.min(dim = 0)[0]
    in_ix_max = in_ix.max(dim = 0)[0]
    
    out_ix = []
    for i in range(int((xmax-xmin)/resolution)):
        for j in range(int((ymax-ymin)/resolution)):
            ix = torch.tensor([i,j])
            print(i,j)
            
            append = True
            
            if i < in_ix_min[0] or i > in_ix_max[0] or j < in_ix_min[1] or j > in_ix_max[1]:
                search = False
            else:
                search = True
            
            if search:
                for k in range(in_ix.size(0)):
                    if ix[0] == in_ix[k,0] and ix[1] == in_ix[k,1]:
                        append = False
                        break
            
            if append:
                out_ix.append(ix.unsqueeze(0))
            
    out_ix = torch.cat(out_ix, dim = 0)
    
    data = index_to_data(out_ix, xmin, ymin, resolution = resolution)
    
    if n_samples > 0:
        out_data = []
        out_data.append(data)
        
        var = resolution / 256
        covar = torch.zeros((2,2))
        covar[0,0], covar[1,1] = var, var
        
        samples = MultivariateNormal(data.float(), covar).sample((n_samples,))
        
        for i in range(n_samples):
            out_data.append(samples[i,:,:])
        out_data = torch.cat(out_data, dim = 0)
        
    else:
        out_data = data
    
    print(out_data.size())
    
    if plot:
        plt.figure()
        plt.plot(maze_data[:,0], maze_data[:,1], 'o', color = '0.4', markersize = 0.5, label = 'in-maze point')
        plt.plot(out_data[:,0], out_data[:,1], 'ok', markersize = 0.5, label = 'out-maze point')
        plt.legend(fontsize = 8, loc = 'upper right')
        plt.xlabel('X-Axis')
        plt.ylabel('Y-Axis')
        plt.title('Generated Out-Maze Data')
        
    return out_data



def create_maze_index_grid(maze_data, xmin, xmax, ymin, ymax, resolution, plot = False):
    
    maze_ix = data_to_index(maze_data, xmin, ymin, resolution).unique(dim = 0)
    
    grid = torch.zeros((int((xmax-xmin)/resolution), int((ymax-ymin)/resolution)))
    #grid[maze_ix] = 1
    
    for i in range(maze_ix.size(0)):
        xix, yix = maze_ix[i,0], maze_ix[i,1]
        grid[xix,yix] = 1
        
    if plot:
        plt.figure()
        img = plt.imshow(grid.transpose(0,1), cmap = 'gray', origin = 'lower')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        #fix x and y axis ticks
        plt.title('Grid Classifier Heat Map')
        plt.colorbar(img)
    
    return grid



def fill_bounds(grid, bounds, xmin, ymin, resolution):

    bounds_ix = data_to_index(bounds, xmin, ymin, resolution)
    
    for i in range(bounds_ix[0,0], bounds_ix[1,0]):
        for j in range(bounds_ix[0,1], bounds_ix[1,1]):
            grid[i,j] = 1
    
    return grid



def create_smoothed_W_manifold_index_grid(
        xmin, xmax, ymin, ymax, resolution = 1, plot = False,
        ):
    
    arm1_bounds = torch.tensor([3, 84, 100, 94]).reshape(2,2)
    arm2_bounds = torch.tensor([3, 5, 25, 84]).reshape(2,2)
    arm3_bounds = torch.tensor([35, 5, 65, 84]).reshape(2,2)
    arm4_bounds = torch.tensor([72, 5, 100, 84]).reshape(2,2)
    
    grid = torch.zeros((int((xmax-xmin)/resolution), int((ymax-ymin)/resolution)))
    
    grid = fill_bounds(grid, arm1_bounds, xmin, ymin, resolution)    
    grid = fill_bounds(grid, arm2_bounds, xmin, ymin, resolution)
    grid = fill_bounds(grid, arm3_bounds, xmin, ymin, resolution)
    grid = fill_bounds(grid, arm4_bounds, xmin, ymin, resolution)
    
    if plot:
        ixs = []
        label = []
        
        for i in range(grid.size(0)):
            for j in range(grid.size(1)):
                ixs.append(torch.tensor([i,j]).unsqueeze(0))
                label.append(grid[i,j].item())
        
        data = index_to_data(torch.cat(ixs, dim = 0), xmin, ymin, resolution)
        #ixs = torch.cat(ixs, dim = 0)
        label = torch.tensor(label)#.unsqueeze(1)
        
        in_ix = torch.nonzero(label)
        in_data = data[in_ix].squeeze()    
        
        maze = torch.load('Datasets/in_data_HL1.pt').squeeze()[:400000]
        plt.figure()
        plt.plot(maze[:,0], maze[:,1], 'o', color = '0.4', markersize = 0.5)
        plt.plot(in_data[:,0], in_data[:,1], 'o', color = 'red', markersize = 0.5)
    
    return grid



def generate_classifier_W_manifold_dataset(xmin, xmax, ymin, ymax, resolution = 1, n_samples = 0):
    
    grid = create_smoothed_W_manifold_index_grid(xmin, xmax, ymin, ymax, resolution)
    
    in_ix = []
    out_ix = []
    
    for i in range(grid.size(0)):
        for j in range(grid.size(1)):
            ix = torch.tensor([i,j]).unsqueeze(0)
            if grid[i,j] == 1:
                in_ix.append(ix)
            else:
                out_ix.append(ix)
            
    in_data = index_to_data(torch.cat(in_ix, dim = 0), xmin, ymin, resolution).float()
    out_data = index_to_data(torch.cat(out_ix, dim = 0), xmin, ymin, resolution).float()
        
    if n_samples == 0:
        return in_data, out_data
    else:
        in_samples = []
        in_samples.append(in_data)
        out_samples = []
        out_samples.append(out_data)
        
        var = resolution**2 / 128
        covar = torch.zeros((2,2))
        covar[0,0], covar[1,1] = var, var
        
        in_samp = MultivariateNormal(in_data, covar).sample((n_samples,))
        out_samp = MultivariateNormal(out_data, covar).sample((n_samples,))
                
        for i in range(n_samples):
            in_samples.append(in_samp[i,:,:])
            out_samples.append(out_samp[i,:,:])
        
        in_samples = torch.cat(in_samples, dim = 0)
        out_samples = torch.cat(out_samples, dim = 0)
        
        print(in_samples.size(), out_samples.size())
    
    return in_samples, out_samples



if __name__ == '__main__':
    
    # boolean toggle indicating whether to generate in-maze data or out-maze data
    gen_in_maze_data = False
    
    
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
        hl = 1
        
        
        # add varying levels of noise to provided in-maze data
        
        new11 = generate_in_maze_data(dat1, variance = 1e-3, history_length = hl, iterations = 2)
        new12 = generate_in_maze_data(dat1, variance = 1e-4, history_length = hl, iterations = 2)
        new13 = generate_in_maze_data(dat1, variance = 1e-5, history_length = hl, iterations = 3)
        new14 = generate_in_maze_data(dat1, variance = 1e-6, history_length = hl, iterations = 1)
        
        new21 = generate_in_maze_data(dat2, variance = 1e-3, history_length = hl, iterations = 2)
        new22 = generate_in_maze_data(dat2, variance = 1e-4, history_length = hl, iterations = 2)
        new23 = generate_in_maze_data(dat2, variance = 1e-5, history_length = hl, iterations = 3)
        new24 = generate_in_maze_data(dat2, variance = 1e-6, history_length = hl, iterations = 1)
        
        new31 = generate_in_maze_data(dat3, variance = 1e-3, history_length = hl, iterations = 2)
        new32 = generate_in_maze_data(dat3, variance = 1e-4, history_length = hl, iterations = 2)
        new33 = generate_in_maze_data(dat3, variance = 1e-5, history_length = hl, iterations = 3)
        new34 = generate_in_maze_data(dat3, variance = 1e-6, history_length = hl, iterations = 1)
    
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
        torch.save(in_data, f'Datasets/in_data_HL{hl}.pt')

    else:
        gen_trajectory = False
        
        if gen_trajectory:
        
            # generate out-maze data
            data = generate_out_maze_data(
                xmin = -50, xmax = 150, ymin = -50, ymax = 150, 
                n_points = 500000, plot = True,
                )
            
            print('\n', data, data.size())
            
            # save out-maze data
            #torch.save(data, 'InMaze/out_trajectory.pt')
            
        else:
            
            in_data = torch.load('Datasets/in_data_HL1.pt').squeeze()
            
            #xmin, xmax, ymin, ymax = -50, 150, -50, 150

# =============================================================================
#             out_data = generate_out_classifier_data(
#                 in_data, xmin, xmax, ymin, ymax, 
#                 resolution = 2, n_samples = 57, plot = True)
#             
#             torch.save(out_data, 'InMaze/out_classifier.pt')
# =============================================================================
            
            xmin, xmax, ymin, ymax = -50, 150, -50, 150

# =============================================================================
#             grid = create_smoothed_W_manifold_index_grid(
#                 xmin, xmax, ymin, ymax, resolution = 1)
#             
#             torch.save(grid, 'InMaze/manifold_grid_classifier.pt')
# =============================================================================

# =============================================================================
#             in_data, out_data = generate_classifier_W_manifold_dataset(
#                 xmin, xmax, ymin, ymax,
#                 resolution = 1, n_samples = 0,)
# =============================================================================
            
            #torch.save(in_data, 'InMaze/manifold_classifier_in_data.pt')
            #torch.save(out_data, 'InMaze/manifold_classifier_out_data.pt')


            grid = create_maze_index_grid(
                in_data, xmin, xmax, ymin, ymax, resolution = 2, plot = True
                )
            print(grid.size())

            #torch.save(grid, 'InMaze/maze_grid_classifier.pt')






