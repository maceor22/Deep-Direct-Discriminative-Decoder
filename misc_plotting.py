import numpy as np
import os
import torch
from copy import deepcopy
from torch.distributions import Normal, MultivariateNormal
from distributions import GaussianMixture, MultivariateNormalMixture
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from maze_utils import Maze, RangeNormalize
from data_generation import data_to_index
import time
import tracemalloc
from memory_profiler import profile

# plotting rat maze data
def plot_spikes_on_maze(maze, single_plot = True):
    
    #ix = maze.spike_counts.sum(0).argsort(descending = True)
    ix = list(range(maze.spike_counts.size(1)))
    pos = []
    for j in ix:
        print(j)
        new = []
        for i in range(maze.spike_counts.size(0)):
            if maze.spike_counts[i,j] > 0:
                new.append(maze.pos[i,:].unsqueeze(0))
        if len(new) != 0:
            pos.append(torch.cat(new, dim = 0))
        print(pos[-1].size())
    
    if single_plot:
        n_row = int(np.floor(len(pos)**0.5))
        n_col = n_row
        while n_row*n_col < len(pos):
            n_col += 1
        
        fig, ax = plt.subplots(nrows = n_row, ncols = n_col, sharex = True, sharey = True)
        fig.suptitle('Spikes over Maze | All Neurons')
        fig.text(0.5, 0.04, 'X-axis', ha = 'center', va = 'center')
        fig.text(0.04, 0.5, 'Y-axis', ha = 'center', va = 'center', rotation = 'vertical')
        for i in range(len(pos)):
            r, c = i // n_col, i % n_col
            print(r,c)
            ax[r,c].plot(maze.pos[:,0], maze.pos[:,1], 'o', color = '0.4', markersize = 2)
            ax[r,c].plot(pos[i][:,0], pos[i][:,1], 'o', color = 'blue', markersize = 1)
        
    else:
        for i in range(len(pos)):
            plt.figure()
            plt.plot(maze.pos[:,0], maze.pos[:,1], 'o', color = '0.4', markersize = 2)
            plt.plot(pos[i][:,0], pos[i][:,1], 'o', color = 'blue', markersize = 4)
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            spike_count = maze.spike_counts[:,ix[i]].sum().long()
            plt.title(f'Spikes over Maze | Neuron {ix[i]+1} | spike_count: {spike_count}')
        

def plot_spikes_versus_regions(region_dict, spikes, positions, save_path):
    
    for c in range(spikes.size(1)):
        print('\ncell', c)
        ix = spikes[:,c].nonzero().squeeze()
        spike = spikes[ix,c]
        pos = positions[ix]

        if ix.dim() == 0:
            pc = 0
            sc = 0
        else:
            pc = ix.size(0)
            sc = spike.sum(0).long().item()

        fig, ax = plt.subplots(ncols = 2, figsize = (16,8))
        fig.suptitle(f'Spike Activity vs Region | Neuron {c} | total points: {pc} | total spikes: {sc}')

        ax[0].plot(
            pos[:,0], pos[:,1], 'o', color = '0.4', ms = 1, 
            label = 'spike locations')
        ax[0].set_xlabel('X-axis', fontsize = 16)
        ax[0].set_ylabel('Y-axis', fontsize = 16)
        ax[0].legend(fontsize = 12, loc = 'lower left')

        point_count = []
        spike_count = []

        for key, reg in region_dict.items():
            mask = pos[:,0] > reg['xmin']
            mask *= pos[:,0] < reg['xmax']
            mask *= pos[:,1] > reg['ymin']
            mask *= pos[:,1] < reg['ymax'] 
            ix = mask.nonzero().squeeze()
            
            if ix.dim() == 0:
                point_count.append(0)
                spike_count.append(0)
            else:
                point_count.append(ix.size(0))
                spike_count.append(spike[ix].sum(0).item())

            ax[0].plot(
                torch.tensor([reg['xmin'], reg['xmax']]),
                torch.tensor([reg['ymin'], reg['ymin']]), 
                'k', lw = 2)
            ax[0].plot(
                torch.tensor([reg['xmin'], reg['xmin']]),
                torch.tensor([reg['ymin'], reg['ymax']]), 
                'k', lw = 2)
            ax[0].plot(
                torch.tensor([reg['xmin'], reg['xmax']]),
                torch.tensor([reg['ymax'], reg['ymax']]), 
                'k', lw = 2)
            ax[0].plot(
                torch.tensor([reg['xmax'], reg['xmax']]),
                torch.tensor([reg['ymin'], reg['ymax']]), 
                'k', lw = 2)
            ax[0].text(
                reg['xmin']+4, reg['ymin']+4, str(key), 
                fontsize = 12, fontweight = 'bold', color = 'red')
        
        barWidth = 0.2
        bar1 = [_ for _ in range(len(region_dict))]
        bar2 = [_ + barWidth for _ in bar1]

        ax[1].bar(bar1, point_count, color = 'orange', width = barWidth,
                  edgecolor = 'grey', label = 'point count')
        ax[1].bar(bar2, spike_count, color = 'blue', width = barWidth,
                  edgecolor = 'grey', label = 'spike count')
        ax[1].legend(fontsize = 12)
        
        ax[1].set_xlabel('Region', fontsize = 16)
        ax[1].set_xticks([_ + 0.5*barWidth for _ in range(len(region_dict))])
        ax[1].set_xticklabels([key for key in region_dict.keys()])

        ax[1].set_ylabel('Counts', fontsize = 16)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        fig.savefig(save_path+f'/neuron{c}.jpeg')
        plt.close(fig)

    plt.show()
        


    return

        
def plot_spike_probs(clf, data, labels):
    norm = RangeNormalize()
    norm.fit(range_min = -50, range_max = 150)
    data = norm.transform(data)
    
    probs = clf.predict(data, return_log_probs = False).transpose(0,1)
    
    
    #plt.pcolormesh(probs)
    fig, ax = plt.subplots()
    ax.imshow(probs, aspect = 'auto', interpolation = 'gaussian',)
    ax.set_xticks(np.arange())



# method for plotting predictions generated by trajectory model
def plot_model_predictions(pred_mean, pred_vars, label_data, title):    
    # pred_mean: prediction mean
    # pred_vars: prediction variance
    # label_data: data used for labels
    # title: title of plot
    # plot_arm_probs: boolean indicating whether to plot arm probability values; only used if arm model is used
    # arm_probs: arm probability values; only used if arm model is used
    
    # convert to numpy
    pred_mean = pred_mean.detach().numpy()
    pred_sdev = pred_vars.detach().numpy()**0.5
    label_data = label_data.detach().numpy()
    
    # time axis
    dt = 0.033
    Time = np.arange(label_data.shape[0]) * dt
    
    # plotting
    fig, ax = plt.subplots(2, 1, sharex = True)
    
    plt.xlabel('Time [s]')
    
    ax[0].plot(Time, label_data[:,0], 'k', label = 'actual')
    ax[0].plot(Time, pred_mean[:,0], 'red', label = 'pred mean', linewidth = 0.7)
    ax[0].fill_between(
        Time,
        pred_mean[:,0] + pred_sdev[:,0],
        pred_mean[:,0] - pred_sdev[:,0],
        color = 'lightsalmon',
        label = 'pred sdev',
        )
    ax[0].legend(fontsize = 6, loc = 2)
    ax[0].set_ylabel('X position')
    
    ax[1].plot(Time, label_data[:,1], 'k', label = 'actual')
    ax[1].plot(Time, pred_mean[:,1], 'red', label = 'pred mean', linewidth = 0.7)
    ax[1].fill_between(
        Time,
        pred_mean[:,1] + pred_sdev[:,1],
        pred_mean[:,1] - pred_sdev[:,1],
        color = 'lightsalmon',
        label = 'pred sdev',
        )
    ax[1].legend(fontsize = 6, loc = 2)
    ax[1].set_ylabel('Y position')
    
    fig.suptitle(title)
    
    return fig



def plot_density_mixture_pi_heatmap(mixture_model, input_data):
    
    pi_k, mu_k, var_k = mixture_model(input_data)
    
    fig, ax = plt.subplots()
    ax.imshow(
        pi_k.T.detach(), aspect = 'auto', 
        interpolation = 'gaussian', origin = 'lower',
        )
    xticks = np.arange(0, pi_k.size(0), 500)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.0f' % i for i in xticks/30])
    yticks = np.arange(0.5, mixture_model.num_mixtures, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticks+0.5).astype('int'))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mixture Probability')
    fig.suptitle('Mixture Probability vs Time')
    

#@profile
def probabilistic_model_likelihood(
        model, model_name, input_data, label_data, posNorm, 
        grid_dim = 100, untransform_label = False, 
        plotting = False, video = False,
    ):
    device = 'cpu'

    model = model.to(device)
    input_data = input_data.to(device)

    latent = torch.linspace(0,1,grid_dim)
    latent = torch.cartesian_prod(latent, latent)
    
    LL = []
    
    xmin, xmax = posNorm.range_min[0].item(), posNorm.range_max[0].item()
    ymin, ymax = posNorm.range_min[1].item(), posNorm.range_max[1].item()
    
    res = (xmax - xmin) / (grid_dim - 1)
    ind = data_to_index(
        posNorm.untransform(label_data.to(device)) if untransform_label else label_data.to(device), 
        xmin, ymin, resolution = res, unique = False,
    )

    n_points = grid_dim**2
    buffer = int(grid_dim/100)
    
    total_likelihood = 0
    
    for i in range(input_data.size(0)):
        p = model.log_prob(input_data[i].expand(n_points,-1,-1), latent).exp().detach()
        p_total = p.sum(dim = 0) 
        
        log_prob = torch.log(p / p_total).view(grid_dim,grid_dim)
        #print(tracemalloc.get_traced_memory())

        xind, yind = ind[i,0], ind[i,1]
        
        total_likelihood += torch.exp(log_prob[xind,yind]) * p_total

        log_prob = torch.where(log_prob < -14, -14, log_prob)
        log_prob[xind-buffer:xind+buffer+1,yind-buffer:yind+buffer+1] = 1

        LL.append(log_prob.detach())

        del p, p_total, log_prob, xind, yind
        #print(tracemalloc.get_traced_memory())
    
    LL = torch.stack(LL, dim = 0)
    
    mean_likelihood = total_likelihood / input_data.size(0)
    
    title = f'{model_name} Marginal Log-Likelihood vs Time'
    title += '\nmean likelihood: %.4f' % mean_likelihood

    if plotting:
        XLL = LL.mean(dim = 2)
        YLL = LL.mean(dim = 1)

        for i in range(ind.size(0)):
            xind, yind = ind[i,0], ind[i,1]
            XLL[i,xind-buffer:xind+buffer+1] = 1
            YLL[i,yind-buffer:yind+buffer+1] = 1

        plot_fig, ax = plt.subplots(nrows = 2, sharex = True)
        
        plot_fig.suptitle(title, fontsize = 10)
        plt.xlabel('Time [s]')
        
        ax[0].pcolormesh(XLL.T, cmap = 'gnuplot2')
        ax[0].set_ylabel('X Position', fontsize = 8)
        xdelta = (xmax-xmin)/6
        ax[0].set_yticks([grid_dim/6,(5*grid_dim)/6])
        ax[0].set_yticklabels([int(xmin+xdelta), int(xmax-xdelta)], fontsize = 8)
        
        ax[1].pcolormesh(YLL.T, cmap = 'gnuplot2')
        ax[1].set_ylabel('Y Position', fontsize = 8)
        ydelta = (ymax-ymin)/6
        ax[1].set_yticks([grid_dim/6,(5*grid_dim)/6])
        ax[1].set_yticklabels([int(ymin+ydelta), int(ymax-ydelta)], fontsize = 8)
        
        xticks = torch.arange(1, 5, 1) * int(label_data.size(0)/5)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels([(i/30).int().item() for i in xticks])
        
    if video:
        images = []

        vid_fig, vid_ax = plt.subplots(nrows = 1)
        vid_fig.suptitle(title, fontsize = 10)
        ticks = np.arange(1,5,1)*int(grid_dim/5)
        vid_ax.set_xticks(ticks)
        vid_ax.set_xticklabels((ticks*res+xmin).astype('int'))
        vid_ax.set_yticks(ticks)
        vid_ax.set_yticklabels((ticks*res+ymin).astype('int'))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        for t in range(input_data.size(0)):
            grid = LL[t]
            
            for i in range(ind.size(0)):
                xind, yind = ind[i,0], ind[i,1]
                if grid[xind,yind] < 1:
                    grid[xind,yind] = -9
            
            img = vid_ax.pcolormesh(grid.T, cmap = 'gnuplot2', animated = True)
            images.append([img])
        vid = animation.ArtistAnimation(
            vid_fig, images, interval = 50, blit = True, repeat_delay = 500, )


    if plotting and video:
        return plot_fig, vid, mean_likelihood
    elif plotting:
        return plot_fig, mean_likelihood
    elif video:
        return vid, mean_likelihood
    else:
        return mean_likelihood


def probabilistic_model_HPD(
        model, model_name, input_data, label_data, posNorm, 
        alpha = 0.1, grid_dim = 100, untransform_label = False, 
        plotting = False, video = False,
    ):
    device = 'cpu'

    model = model.to(device)
    input_data = input_data.to(device)

    latent = torch.linspace(0,1,grid_dim)
    latent = torch.cartesian_prod(latent, latent)
    
    HPD = []
    
    xmin, xmax = posNorm.range_min[0].item(), posNorm.range_max[0].item()
    ymin, ymax = posNorm.range_min[1].item(), posNorm.range_max[1].item()
    
    res = (xmax - xmin) / (grid_dim - 1)
    ind = data_to_index(
        posNorm.untransform(label_data.to(device)) if untransform_label else label_data.to(device), 
        xmin, ymin, resolution = res, unique = False,
    )

    xmin, ymin = posNorm.range_min[0], posNorm.range_min[1]
    xmax, ymax = posNorm.range_max[0], posNorm.range_max[1]

    n_points = grid_dim**2
    buffer = int(grid_dim/100)
    
    area_ratio = 0
    accuracy = 0

    #tracemalloc.start()
    
    for i in range(input_data.size(0)):
        p = model.log_prob(input_data[i].expand(n_points,-1,-1), latent).exp().detach()
        p /= p.sum(dim = 0)
        
        sorted, ix = p.sort(dim = 0, descending = True)

        idx = 0
        while sorted[:idx].sum(0) < (1 - alpha):
            idx += 1
        
        hpd = torch.ones((n_points,))
        hpd[ix[:idx]] = 0.4
        hpd = hpd.view(grid_dim,grid_dim)

        area_ratio += idx

        xind, yind = ind[i,0], ind[i,1]
        if hpd[xind,yind] != 1:
            accuracy += 1
        
        hpd[xind-buffer:xind+buffer+1,yind-buffer:yind+buffer+1] = 0

        HPD.append(hpd)

        del p, xind, yind
        #print(tracemalloc.get_traced_memory())
    
    HPD = torch.stack(HPD, dim = 0)
    
    area_ratio /= (input_data.size(0) * n_points)
    accuracy /= input_data.size(0)
    
    title = f'{model_name} {int((1-alpha)*100)}% Highest Posterior Density Region (HDR) vs Time'
    title += '\nratio samples in HDR: %.4f' % accuracy
    title += ' | mean HDR area coverage ratio: %.4f' % area_ratio

    if plotting:
        xtemp = HPD.sum(dim = 2)
        ytemp = HPD.sum(dim = 1)

        X_likelihood = torch.where(xtemp < grid_dim, 0.4, 1)
        Y_likelihood = torch.where(ytemp < grid_dim, 0.4, 1)

        for i in range(ind.size(0)):
            xind, yind = ind[i,0], ind[i,1]
            X_likelihood[i,xind-buffer:xind+buffer+1] = 0
            Y_likelihood[i,yind-buffer:yind+buffer+1] = 0

        plot_fig, ax = plt.subplots(nrows = 2, sharex = True)
        
        plot_fig.suptitle(title, fontsize = 10)
        plt.xlabel('Time [s]')
        
        ax[0].pcolormesh(X_likelihood.T.detach(), cmap = 'gist_stern')
        ax[0].set_ylabel('X Position', fontsize = 8)
        xdelta = (xmax-xmin)/6
        ax[0].set_yticks([grid_dim/6,(5*grid_dim)/6])
        ax[0].set_yticklabels([int(xmin+xdelta), int(xmax-xdelta)], fontsize = 8)
        
        ax[1].pcolormesh(Y_likelihood.T.detach(), cmap = 'gist_stern')
        ax[1].set_ylabel('Y Position', fontsize = 8)
        ydelta = (ymax-ymin)/6
        ax[1].set_yticks([grid_dim/6,(5*grid_dim)/6])
        ax[1].set_yticklabels([int(ymin+ydelta), int(ymax-ydelta)], fontsize = 8)
        
        xticks = torch.arange(1, 5, 1) * int(input_data.size(0)/5)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels([(i/30).int().item() for i in xticks])
        
    if video:
        images = []

        vid_fig, vid_ax = plt.subplots(nrows = 1)
        vid_fig.suptitle(title, fontsize = 10)
        ticks = np.arange(1,5,1)*int(grid_dim/5)
        vid_ax.set_xticks(ticks)
        vid_ax.set_xticklabels((ticks*res+xmin.item()).astype('int'))
        vid_ax.set_yticks(ticks)
        vid_ax.set_yticklabels((ticks*res+ymin.item()).astype('int'))
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        for t in range(input_data.size(0)):
            grid = HPD[t]
            
            for i in range(ind.size(0)):
                xind, yind = ind[i,0], ind[i,1]
                if grid[xind,yind] > 0:
                    grid[xind,yind] = 0.6
            
            img = vid_ax.pcolormesh(grid.T, cmap = 'gist_stern', animated = True)
            images.append([img])
        vid = animation.ArtistAnimation(
            vid_fig, images, interval = 50, blit = True, repeat_delay = 500, )


    if plotting and video:
        return plot_fig, vid, accuracy, area_ratio
    elif plotting:
        return plot_fig, accuracy, area_ratio
    elif video:
        return vid, accuracy, area_ratio
    else:
        return accuracy, area_ratio



def plot_density_mixture_metrics(root):
    history_lengths = torch.load(root+'/metrics/history_lengths.pt').numpy()

    #train_acc = torch.load(root+'/metrics/train_accuracy.pt')
    #train_area = torch.load(root+'/metrics/train_area.pt')
    #train_score = torch.load(root+'/metrics/train_ratio.pt')
    #train_mse = torch.load(root+'/metrics/train_mse.pt')

    test_acc = torch.load(root+'/metrics/test_accuracy.pt')
    test_area = torch.load(root+'/metrics/test_area.pt')
    test_score = torch.load(root+'/metrics/test_ratio.pt')
    test_mse = torch.load(root+'/metrics/test_mse.pt')


    #tr_acc_mean = train_acc.mean(dim = 1)
    #tr_area_mean = train_area.mean(dim = 1)
    #tr_score_mean = train_score.mean(dim = 1)
    #tr_mse_mean = train_mse.mean(dim = 1)

    #tr_acc_std = train_acc.std(dim = 1)
    #tr_area_std = train_area.std(dim = 1)
    #tr_score_std = train_score.std(dim = 1)
    #tr_mse_std = train_mse.std(dim = 1)


    te_acc_mean = test_acc.mean(dim = 1)
    te_area_mean = test_area.mean(dim = 1)
    te_score_mean = test_score.mean(dim = 1)
    te_mse_mean = test_mse.mean(dim = 1)

    te_acc_std = test_acc.std(dim = 1)
    te_area_std = test_area.std(dim = 1)
    te_score_std = test_score.std(dim = 1)
    te_mse_std = test_mse.std(dim = 1)
    
    fig, ax = plt.subplots(nrows = 3, sharex = True)
    fontsize = 6
    """ ax[0].plot(
        history_lengths-history_lengths[0], tr_acc_mean, 
        '0.2', label = 'train mean',
        )
    ax[0].fill_between(
        history_lengths-history_lengths[0], tr_acc_mean + tr_acc_std, tr_acc_mean - tr_acc_std,
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[0].plot(
        history_lengths-history_lengths[0], te_acc_mean, 
        'navy', label = 'test mean',
    )
    ax[0].fill_between(
        history_lengths-history_lengths[0], 
        te_acc_mean + te_acc_std, te_acc_mean - te_acc_std,
        color = 'lightskyblue', label = 'test sdev', alpha = 0.4,
    )
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(fontsize = fontsize)

    """ ax[1].plot(
        history_lengths-history_lengths[0], tr_area_mean, 
        '0.2', label = 'train mean',
        )
    ax[1].fill_between(
        history_lengths-history_lengths[0], 
        tr_area_mean + tr_area_std, tr_area_mean - tr_area_std,
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[1].plot(
        history_lengths-history_lengths[0], te_area_mean, 
        'darkorange', label = 'test mean',
    )
    ax[1].fill_between(
        history_lengths-history_lengths[0], 
        te_area_mean + te_area_std, te_area_mean - te_area_std,
        color = 'bisque', label = 'test sdev', alpha = 0.4,
    )
    ax[1].set_ylabel('Area')
    ax[1].legend(fontsize = fontsize)
    
    """ ax[2].plot(
        history_lengths-history_lengths[0], tr_score_mean, 
        '0.2', label = 'train mean',
        )
    ax[2].fill_between(
        history_lengths-history_lengths[0], 
        tr_score_mean + tr_score_std, tr_score_mean - tr_score_std,
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[2].plot(
        history_lengths-history_lengths[0], te_score_mean, 
        'darkgreen', label = 'test mean',
        )
    ax[2].fill_between(
        history_lengths-history_lengths[0], 
        te_score_mean + te_score_std, te_score_mean - te_score_std,
        color = 'lime', label = 'test sdev', alpha = 0.4,
    )
    ax[2].set_ylabel('Ratio')
    ax[2].legend(fontsize = fontsize)
    xticks = np.arange(
        0, history_lengths[-1]+1-history_lengths[0], 
        step = history_lengths[1] - history_lengths[0])
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticks+history_lengths[0])

    plt.xlabel('History Length')
    fig.suptitle(f'Effect of History Length on HPD Metrics\n{test_acc.size(1)}-Fold Cross Validation')
    
    fig.savefig(root+'/HPD_results.jpeg')
    plt.close(fig)


    fig, ax = plt.subplots(nrows = 3, sharex = True)

    """ ax[0].plot(
        history_lengths-history_lengths[0], tr_mse_mean[:,0], 
        '0.2', label = 'train mean',
        )
    ax[0].fill_between(
        history_lengths-history_lengths[0], 
        tr_mse_mean[:,0] + tr_mse_std[:,0], tr_mse_mean[:,0] - tr_mse_std[:,0],
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[0].plot(
        history_lengths-history_lengths[0], te_mse_mean[:,0], 
        'navy', label = 'test mean',
    )
    ax[0].fill_between(
        history_lengths-history_lengths[0], 
        te_mse_mean[:,0] + te_mse_std[:,0], te_mse_mean[:,0] - te_mse_std[:,0],
        color = 'lightskyblue', label = 'test sdev', alpha = 0.4,
    )
    ax[0].set_ylabel('Average MSE')
    ax[0].legend(fontsize = fontsize)

    """ ax[1].plot(
        history_lengths-history_lengths[0], tr_mse_mean[:,1], 
        '0.2', label = 'train mean',
        )
    ax[1].fill_between(
        history_lengths-history_lengths[0], 
        tr_mse_mean[:,1] + tr_mse_std[:,1], tr_mse_mean[:,1] - tr_mse_std[:,1],
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[1].plot(
        history_lengths-history_lengths[0], te_mse_mean[:,1], 
        'darkorange', label = 'test mean',
    )
    ax[1].fill_between(
        history_lengths-history_lengths[0], 
        te_mse_mean[:,1] + te_mse_std[:,1], te_mse_mean[:,1] - te_mse_std[:,1],
        color = 'bisque', label = 'test sdev', alpha = 0.4,
    )
    ax[1].set_ylabel('Dominant MSE')
    ax[1].legend(fontsize = fontsize)
    
    """ ax[2].plot(
        history_lengths-history_lengths[0], tr_mse_mean[:,2], 
        '0.2', label = 'train mean',
        )
    ax[2].fill_between(
        history_lengths-history_lengths[0], 
        tr_mse_mean[:,2] + tr_mse_std[:,2], tr_mse_mean[:,2] - tr_mse_std[:,2],
        color = '0.6', label = 'train sdev', alpha = 0.8,
    ) """
    ax[2].plot(
        history_lengths-history_lengths[0], te_mse_mean[:,2], 
        'darkgreen', label = 'test mean',
    )
    ax[2].fill_between(
        history_lengths-history_lengths[0], 
        te_mse_mean[:,2] + te_mse_std[:,2], te_mse_mean[:,2] - te_mse_std[:,2],
        color = 'lime', label = 'test sdev', alpha = 0.4,
    )
    ax[2].set_ylabel('Sample MSE')
    ax[2].legend(fontsize = fontsize)

    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticks+history_lengths[0])

    plt.xlabel('History Length')
    fig.suptitle(f'Effect of History Length on Mean Squared Error\n{test_acc.size(1)}-Fold Cross Validation')
    
    fig.savefig(root+'/MSE_results.jpeg')
    plt.close(fig)

    return


def raster_plot(spikes, pos):
    fig, ax = plt.subplots(
        nrows = 3, sharex = True,
        gridspec_kw = {'height_ratios' : [2,1,1]})

    fig.suptitle('Raster Plot')

    #ax.imshow(spikes, aspect = 'auto', interpolation = 'none', origin = 'lower')
    ax[0].pcolormesh(spikes, cmap = 'Greys')
    ax[0].set_ylabel('Place Cells')

    ax[1].plot(range(pos.size(0)), pos[:,0], 'k')
    ax[1].set_ylabel('X Position')

    ax[2].plot(range(pos.size(0)), pos[:,1], 'k')
    ax[2].set_ylabel('Y Position')

    xticks = ax[2].get_xticks()
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels((xticks/30).astype('int'))
    ax[2].set_xlabel('Time [s]')

    return fig


def plot_maze(maze_data):
    fig, ax = plt.subplots()

    fig.suptitle('Maze Structure Top View')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.plot(maze_data[:,0], maze_data[:,1], 'o', color = '0.4', ms = 2)

    return fig


def plot_pretrained_learned_mixtures(mixture_model, input_data, label_data, posNorm):
    mixture_model.to(input_data.device)
    distribution = mixture_model(input_data)

    sample = distribution.component_distribution.base_dist.sample()
    
    sample = posNorm.untransform(sample)
    label_data = posNorm.untransform(label_data)
    
    fig, ax = plt.subplots()
    ax.plot(label_data[:,0], label_data[:,1], 'o', color = '0.4', label = 'maze', markersize = 0.5)
    for k in range(mixture_model.num_mixtures):
        ax.plot(sample[:,k,0], sample[:,k,1], 'o', label = f'mixture {k+1}', markersize = 2)
    ax.legend(fontsize = 8)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.suptitle('Learned Mixtures from Pre-trained Model')

    return fig
    


def plot_filter_performance(D4, latent_labels, transform, rmse):
    PARTICLES = []
    for i in range(latent_labels.size(0)):
        PARTICLES.append(D4.u_particles[i])
        PARTICLES.append(D4.w_particles[i])

    fig, ax = plt.subplots(nrows = 1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    xmin, ymin = transform.range_min[0], transform.range_min[1]
    xmax, ymax = transform.range_max[0], transform.range_max[1]
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    fig.suptitle(f'D4 Filter Performance | RMSE: %.3f\nnumber of particles: {D4.n_particles}' % rmse)
    
    ax.plot(latent_labels[:,0], latent_labels[:,1], 'o', color = '0.4', markersize = 0.5)

    """ particle_traces = [
        ax.plot([], [], color = 'cyan', linewidth = 1)[0] for _ in range(D4.n_particles)
        ] """

    particles, = ax.plot([], [], 'o', color = 'blue', markersize = 5)

    """ latent_trace, = ax.plot([], [], color = 'k', linewidth = 3) """
    latent_point, = ax.plot([], [], 'o', color = 'k', markersize = 15)

    time_text = ax.text(
        xmin+5, ymax-15, '', fontsize = 10)

    def animator(i):
        t = int(i/2)

        """ trace_data = transform.untransform(D4.traces[i].detach()) """
        particle_data = transform.untransform(PARTICLES[i].detach())

        """ for n in range(D4.n_particles): """
        """     particle_traces[n].set_data(trace_data[n,:,0], trace_data[n,:,1]) """
        
        particles.set_data(particle_data[:,0], particle_data[:,1])
        
        """ latent_trace.set_data(latent_labels[:t+1,0], latent_labels[:t+1,1]) """
        latent_point.set_data(latent_labels[t,0], latent_labels[t,1])

        time_text.set_text('time = %.3f' % (t*0.033))
        
        """ return [*particle_traces, particles, latent_trace, latent_point, time_text] """
        return [particles, latent_point, time_text]

    vid = animation.FuncAnimation(
        fig, animator, 2*latent_labels.size(0), interval = 25, blit = True, repeat_delay = 500,
    )
    return vid


def plot_filter_performance1(D4, latent_labels, transform):
    fig, ax = plt.subplots(nrows = 1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    fig.suptitle(f'D4 Filter Performance\nnumber of particles: {D4.n_particles}')
    
    ax.plot(latent_labels[:,0], latent_labels[:,1], 'o', color = '0.4', markersize = 0.5)

    particle_traces = [
        ax.plot([], [], color = 'cyan', linewidth = 1)[0] for _ in range(D4.n_particles)
        ]

    particles, = ax.plot([], [], 'o', color = 'blue', markersize = 5)

    latent_trace, = ax.plot([], [], color = 'k', linewidth = 3)
    latent_point, = ax.plot([], [], 'o', color = 'k', markersize = 15)

    time_text = ax.text(
        latent_labels.min(0)[0][0]+2, latent_labels.max(0)[0][1]-2, '', fontsize = 10)

    images = []

    for i in range(2*latent_labels.size(0)):
        t = int(i/2)
        print(t)

        trace_data = transform.untransform(D4.traces[i].detach())
        particle_data = transform.untransform(D4.particles[i].detach())

        for n in range(D4.n_particles):
            particle_traces[n].set_data(trace_data[n,:,0], trace_data[n,:,1])
        
        particles.set_data(particle_data[:,0], particle_data[:,1])
        
        latent_trace.set_data(latent_labels[:t+1,0], latent_labels[:t+1,1])
        latent_point.set_data(latent_labels[t,0], latent_labels[t,1])

        time_text.set_text('time = %.3f' % (t*0.033))
        
        images.append([*particle_traces, particles, latent_point, time_text])

    vid = animation.ArtistAnimation(
        fig, images, interval = 300, blit = True, repeat_delay = 1000,
    )
    return vid



def filter_video(self, labels):
        fig, ax = plt.subplots()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        images = []

        unweighted = self.unweighted[0].detach()
        weighted = self.weighted[0].detach()

        img1 = ax.plot(labels[:,0], labels[:,1], 'o', color = '0.4', markersize = 0.2)
        ax.plot(unweighted[:,0], unweighted[:,1], 'o', color = 'cyan', markersize = 0.5)
        ax.plot(labels[0,0], labels[0,1], 'o', color = 'k', markersize = 1)
        images.append([img1])

        img2 = ax.plot(labels[:,0], labels[:,1], 'o', color = '0.4', markersize = 0.2)
        ax.plot(weighted[:,0], weighted[:,1], 'o', color = 'blue', markersize = 0.5)
        ax.plot(labels[0,0], labels[0,1], 'o', color = 'k', markersize = 1)
        images.append([img2])

        for t in range(1, labels.size(0)):
            print(t)
            unweighted_particles = self.unweighted[t].detach()
            unweighted_traces = self.traces[t-1].detach()

            weighted_particles = self.weighted[t].detach()
            weighted_traces = self.traces[t].detach()

            img1 = ax.plot(labels[:,0], labels[:,1], 'o', color = '0.4', markersize = 0.2)
            for n in range(self.n_particles):
                ax.plot(
                    unweighted_traces[n,:,0], unweighted_traces[n,:,1], 
                    color = 'cyan', linewidth = 0.1,
                    )
            ax.plot(
                unweighted_particles[:,0], unweighted_particles[:,1], 
                'o', color = 'cyan', markersize = 0.5,
                )
            ax.plot(labels[:t,0], labels[:t,1], color = 'k', linewidth = 0.2)
            ax.plot(labels[t,0], labels[t,1], 'o', color = 'k', markersize = 1)
            images.append([img1])

            img2 = ax.plot(labels[:,0], labels[:,1], 'o', color = '0.4', markersize = 0.2)
            for n in range(self.n_particles):
                ax.plot(
                    weighted_traces[n,:,0], weighted_traces[n,:,1], 
                    color = 'cyan', linewidth = 0.1,
                    )
            ax.plot(
                weighted_particles[:,0], weighted_particles[:,1], 
                'o', color = 'blue', markersize = 0.5,
                )
            ax.plot(labels[:t,0], labels[:t,1], color = 'k', linewidth = 0.2)
            ax.plot(labels[t,0], labels[t,1], 'o', color = 'k', markersize = 1)
            images.append([img2])

        vid = animation.ArtistAnimation(
            fig, images, interval = 100, blit = True, repeat_delay = 500, )
        
        return vid










