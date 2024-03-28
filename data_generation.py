import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.distributions.normal import Normal
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
    for i in range(history_length+1, data.size(dim = 0)):
        # generate trajectory history and append to list
        new = data[i-(history_length+1):i,:,:].transpose(0,-1).transpose(1,-1)
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
    
    xstep = int((xmax-xmin)/resolution)
    ystep = int((ymax-ymin)/resolution)
    grid = torch.zeros((xstep, ystep))
    
    for i in range(maze_ix.size(0)):
        xix, yix = maze_ix[i,0], maze_ix[i,1]
        grid[xix,yix] = 1
        
    if plot:
        fig, ax = plt.subplots()
        img = ax.imshow(grid.transpose(0,1), cmap = 'gray', origin = 'lower')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ticks = np.arange(1,4)
        ax.set_xticks(ticks*xstep/4)
        ax.set_yticks(ticks*ystep/4)
        ax.set_xticklabels(xmin + ticks*xstep)
        ax.set_yticklabels(ymin + ticks*ystep)
        #fix x and y axis ticks
        fig.suptitle('Grid Classifier Heat Map')
        plt.colorbar(img)
        plt.show()
    
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



def balance_dataset(
        label_data, xmin, xmax, ymin, ymax, resolution,
        threshold = None, presence = False, p_range = 1, 
        delta_included = False, dmin = None, dmax = None,
        down_sample = True, up_sample = False, plotting = False,
        ):
    
    label_data = label_data.unsqueeze(1) if label_data.dim() == 2 else label_data
    
    if delta_included:
        if dmin == None:
            dmin = -1
        if dmax == None:
            dmax = 1e3
        ixs = []
        for i in range(label_data.size(0)):
            inf_norm = label_data[i,-1,2:].abs().max(0)[0]
            if inf_norm > dmin and inf_norm < dmax:
                ixs.append(i)
        ixs = torch.tensor(ixs).long()
        label_data = label_data[ixs]

    grid_list = []
    for i in range(int((xmax-xmin)/resolution)):
        grid_list.append([])
        for j in range(int((ymax-ymin)/resolution)):
            grid_list[i].append([])
    
    grid_ix = data_to_index(label_data[:,-1,:2], xmin, ymin, resolution, unique = False)
    
    for i in range(grid_ix.size(0)):
        grid_list[grid_ix[i,0]][grid_ix[i,1]].append(i)
    
    grid_tensor = torch.zeros((len(grid_list), len(grid_list[0])))
    for i in range(grid_tensor.size(0)):
        for j in range(grid_tensor.size(1)):
            grid_tensor[i,j] = len(grid_list[i][j])
    
    if presence:
        p_tensor = torch.zeros_like(grid_tensor)
        for i in range(p_range, p_tensor.size(0)-p_range):
            for j in range(p_range, p_tensor.size(1)-p_range):
                p_tensor[i,j] = grid_tensor[
                    i-p_range:i+p_range+1,j-p_range:j+p_range+1
                    ].sum(dim = 0).sum(dim = 0)
        threshold = p_tensor.flatten().max(dim = 0)[0].long().item()
        p_min = p_tensor.flatten().sort()[0].unique()[1].item()
        p_tensor = p_min / p_tensor
        p_tensor[torch.isinf(p_tensor)] = 0
        
    if threshold == None and not presence:
        threshold = 1000
    
    data = torch.zeros((threshold,))
    indices = []
    
    for i in range(len(grid_list)):
        for j in range(len(grid_list[i])):
            if presence:
                density = p_tensor[i,j]
                data_ix = (p_min/density).long()-1
                if down_sample:
                    sample_size = (density*grid_tensor[i,j]).ceil().long()
                
            else:
                density = grid_tensor[i,j].long()
                data_ix = density-1
                if down_sample:
                    sample_size = min(len(grid_list[i][j]), threshold)

            if density > 0:
                if up_sample and presence:
                    n_upsamples = ((density+1e-9)**-1 * grid_tensor[i,j]).ceil().long().item()
                elif up_sample:
                    n_upsamples = max(1, int(threshold // grid_tensor[i,j]))

                if density <= threshold: data[data_ix] += 1
                ixs = torch.tensor(grid_list[i][j])
                if down_sample:
                    ixs = ixs[torch.randperm(ixs.size(0))][:sample_size]
                if up_sample:
                    ixs = ixs.repeat(n_upsamples)
                indices.append(ixs)
    
    indices = torch.cat(indices, dim = 0).long()
    
    if plotting:
        xaxis = torch.arange(1,threshold+1).float()
        if presence:
            ixs = data.nonzero(as_tuple = True)
            data = data[ixs]
            xaxis = xaxis[ixs]
            xlabel_add = 'presense ratio'
            title_add = 'presense score'
            xticklabels = ['%.1f' % i for i in np.arange(start = 0, stop = 1.1, step = .1)]
        else:
            xlabel_add = f'per {resolution**2} cm^2'
            title_add = 'concentration score'
            xticklabels = ['%d' % i for i in np.arange(start = 0, stop = threshold+1, step = threshold/10)]
        
        fig, ax = plt.subplots()
        ax.plot(xaxis, data, color = 'k')
        ax.set_xticks(np.arange(start = 0, stop = threshold+1, step = threshold/10))
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(f'Density of Data Points [{xlabel_add}]')
        ax.set_ylabel('Number of Locations')
        fig.suptitle(f'Density of Data Points vs Number of Locations\n{title_add} | resolution: {resolution} cm')
        
        
        if presence:
            ixs = p_tensor.nonzero()
            for i in range(ixs.size(0)):
                p_tensor[ixs[i,0], ixs[i,1]] = p_tensor[ixs[i,0], ixs[i,1]]**-1
            data = p_tensor
            cmap = 'magma'
        else:
            data = grid_tensor
            cmap = 'inferno'
        
        fig, ax = plt.subplots()
        ax.imshow(
            data.transpose(0,1), cmap = cmap, 
            interpolation = 'gaussian', norm = None, 
            aspect = 'auto', origin = 'lower',
            )
        ax.set_xticks(np.arange(2,11,2)*(10/resolution))
        ax.set_xticklabels(np.arange(20,101,20) + xmin)
        ax.set_yticks(np.arange(2,11,2)*(10/resolution))
        ax.set_yticklabels(np.arange(20,101,20) + ymin)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        fig.suptitle(f'Density Heatmap\n{title_add} | resolution: {resolution} cm')
        
    return indices



def generate_dataset(
        rat_name, pos_history_length, spike_history_length, spike_bin_size, 
        label_history_length, include_velocity = False, dirvel = True, 
        grid_resolution = 5, balance_resolution = 1, threshold = None, 
        presence = False, p_range = 1,
        delta_included = False, dmin = None, dmax = None,
        down_sample = True, up_sample = False,
        ):
    
    if rat_name == 'Bon':
        xmin, xmax, ymin, ymax = 150, 270, 50, 170
    elif rat_name == 'Emile':
        xmin, xmax, ymin, ymax = -10, 260, -10, 260
    
    path = 'Datasets/data/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        rat_name, pos_history_length, spike_history_length, spike_bin_size, 
        label_history_length, include_velocity, dirvel, 
        grid_resolution, balance_resolution, threshold, 
        presence, p_range, delta_included, dmin, dmax, down_sample, up_sample,
        )
    
    if os.path.exists(path):
        print('loading datasets.....')
        grid = torch.load(path + '/maze_grid.pt')
        bal_train_spikes = torch.load(path + '/bal_train_spikes.pt')
        bal_train_pos = torch.load(path + '/bal_train_positions.pt')
        bal_train_label = torch.load(path + '/bal_train_label.pt')
        raw_train_spikes = torch.load(path + '/raw_train_spikes.pt')
        raw_train_pos = torch.load(path + '/raw_train_positions.pt')
        raw_train_label = torch.load(path + '/raw_train_label.pt')
        raw_valid_spikes = torch.load(path + '/raw_valid_spikes.pt')
        raw_valid_pos = torch.load(path + '/raw_valid_positions.pt')
        raw_valid_label = torch.load(path + '/raw_valid_label.pt')
        raw_test_spikes = torch.load(path + '/raw_test_spikes.pt')
        raw_test_pos = torch.load(path + '/raw_test_positions.pt')
        raw_test_label = torch.load(path + '/raw_test_label.pt')
        
    else:
    
        print('\nloading raw data.....')
        
        if rat_name == 'Bon':
            
            wm1 = Maze(
                name = 'Bon', 
                session = (3,1), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm2 = Maze(
                name = 'Bon', 
                session = (3,3), 
                n_points = 'all',
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            
            mazes = (wm1, wm2)
            rem_cells = []
        
        
        elif rat_name == 'Emile':
            
            wm1 = Maze(
                name = 'Emile', 
                session = (20,1), 
                n_points = 'all',
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm1.pos = wm1.pos[1000:]
            wm1.spike_counts = wm1.spike_counts[1000:]
            #wm1.plot_position(plot_map = True, plot_over_time = True)

            wm2 = Maze(
                name = 'Emile', 
                session = (20,3), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm2.pos = wm2.pos[1000:]
            wm2.spike_counts = wm2.spike_counts[1000:]
            #wm2.plot_position(plot_map = True, plot_over_time = True)

            wm3 = Maze(
                name = 'Emile', 
                session = (20,4), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm3.pos = wm3.pos[1000:]
            wm3.spike_counts = wm3.spike_counts[1000:]
            #wm3.plot_position(plot_map = True, plot_over_time = True)

            wmtemp = Maze(
                name = 'Emile', 
                session = (20,6), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            
            wm4 = deepcopy(wmtemp)
            wm4.pos = wm4.pos[1000:75000]
            wm4.spike_counts = wm4.spike_counts[1000:75000]
            #wm4.plot_position(plot_map = True, plot_over_time = True)

            wm5 = deepcopy(wmtemp)
            wm5.pos = wm5.pos[84000:]
            wm5.spike_counts = wm5.spike_counts[84000:]
            #wm5.plot_position(plot_map = True, plot_over_time = True)

            mazes = (wm1, wm2, wm3, wm4, wm5)

            rem_cells = [
                11, 14, 24, 26, 27, 28, 29, 31, 32, 33, 
                34, 35, 37, 38, 41, 43, 46, 47, 48, 52, 
                53, 54, 56, 58, 59, 60, 61, 66, 68, 72, 
                75, 76, 78, 81, 83, 85, 89, 
                ]
            
        
        print('\ngenerating history terms.....')
        spikes, positions, labels = [], [], []
        for m in mazes:
            new_spikes, new_positions, new_labels = m.generate_data(
                pos_history_length = pos_history_length, 
                spike_history_length = spike_history_length, spike_bin_size = spike_bin_size, 
                label_history_length = label_history_length, shuffle = False,
                )
            
            spikes.append(new_spikes)
            positions.append(new_positions)
            labels.append(new_labels)

        spikes = torch.cat(spikes, dim = 0)
        spikes = torch.stack(
            [spikes[...,c] for c in range(spikes.size(2)) if c not in rem_cells], 
            dim = 2,
            )

        positions = torch.cat(positions, dim = 0)
        labels = torch.cat(labels, dim = 0)

        if rat_name == 'Bon':
            b0 = int(-6000)
            b1 = int(-3000)
        elif rat_name == 'Emile':
            b0 = int(labels.size(0)*0.85)
            b1 = int(labels.size(0)*0.97)

        """ train_spikes, raw_test_spikes = torch.cat(spikes[:-1], dim = 0), spikes[-1]
        train_positions, raw_test_pos = torch.cat(positions[:-1], dim = 0), positions[-1]
        train_labels, raw_test_label = torch.cat(labels[:-1], dim = 0), labels[-1]

        b = int(train_labels.size(0)*0.85) """
        
        print('\ngenerating maze grid.....')
        grid = create_maze_index_grid(
            maze_data = labels[:b0,:2] if labels.dim() == 2 else labels[:b0,-1,:2],
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = grid_resolution, plot = False,
        )

        """ grid = create_maze_index_grid(
            maze_data = train_labels[:b,:2] if train_labels.dim() == 2 else train_labels[:b,-1,:2],
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = grid_resolution, plot = True,
        ) """

        print('\nbalancing training data.....')
        bal_ix = balance_dataset(
            label_data = torch.cat([labels[:b0,:2], labels[:b0,:2]-positions[:b0,-1,:2]], dim = 1), 
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = balance_resolution, threshold = threshold, 
            presence = presence, p_range = p_range,
            delta_included = True, dmin = dmin, dmax = dmax, 
            down_sample = down_sample, up_sample = up_sample, plotting = False,
            )
        
        """ bal_ix = balance_dataset(
            label_data = train_labels[:b], xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = balance_resolution, threshold = threshold, 
            presence = presence, p_range = p_range, 
            delta_included = include_velocity, dmin = dmin, dmax = dmax, plotting = False,
            ) """
        
        bal_train_spikes, bal_train_pos, bal_train_label = spikes[bal_ix], positions[bal_ix], labels[bal_ix]
        raw_train_spikes, raw_train_pos, raw_train_label = spikes[:b0], positions[:b0], labels[:b0]
        raw_valid_spikes, raw_valid_pos, raw_valid_label = spikes[b0:b1], positions[b0:b1], labels[b0:b1]
        raw_test_spikes, raw_test_pos, raw_test_label = spikes[b1:], positions[b1:], labels[b1:]

        """ bal_train_spikes, bal_train_pos, bal_train_label = train_spikes[bal_ix], train_positions[bal_ix], train_labels[bal_ix]
        raw_valid_spikes, raw_valid_pos, raw_valid_label = train_spikes[b:], train_positions[b:], train_labels[b:]
         """
        
        if not down_sample:
            bal_train_pos += Normal(
                loc = torch.zeros_like(bal_train_pos), 
                scale = torch.ones_like(bal_train_pos)*1e-6,
                ).sample()
            bal_train_label += Normal(
                loc = torch.zeros_like(bal_train_label), 
                scale = torch.ones_like(bal_train_label)*1e-3,
                ).sample()
        
        os.mkdir(path)
        
        torch.save(grid, path + '/maze_grid.pt')
        torch.save(bal_train_spikes, path + '/bal_train_spikes.pt')
        torch.save(bal_train_pos, path + '/bal_train_positions.pt')
        torch.save(bal_train_label, path + '/bal_train_label.pt')
        torch.save(raw_train_spikes, path + '/raw_train_spikes.pt')
        torch.save(raw_train_pos, path + '/raw_train_positions.pt')
        torch.save(raw_train_label, path + '/raw_train_label.pt')
        torch.save(raw_valid_spikes, path + '/raw_valid_spikes.pt')
        torch.save(raw_valid_pos, path + '/raw_valid_positions.pt')
        torch.save(raw_valid_label, path + '/raw_valid_label.pt')
        torch.save(raw_test_spikes, path + '/raw_test_spikes.pt')
        torch.save(raw_test_pos, path + '/raw_test_positions.pt')
        torch.save(raw_test_label, path + '/raw_test_label.pt')
    
    ret = {}
    ret['maze_grid'] = grid
    ret['bal_train_spikes'] = bal_train_spikes
    ret['bal_train_positions'] = bal_train_pos
    ret['bal_train_labels'] = bal_train_label
    ret['raw_train_spikes'] = raw_train_spikes
    ret['raw_train_positions'] = raw_train_pos
    ret['raw_train_labels'] = raw_train_label
    ret['raw_valid_spikes'] = raw_valid_spikes
    ret['raw_valid_positions'] = raw_valid_pos
    ret['raw_valid_labels'] = raw_valid_label
    ret['raw_test_spikes'] = raw_test_spikes
    ret['raw_test_positions'] = raw_test_pos
    ret['raw_test_labels'] = raw_test_label
    ret['xmin'], ret['xmax'], ret['ymin'], ret['ymax'] = xmin, xmax, ymin, ymax
    
    return ret



