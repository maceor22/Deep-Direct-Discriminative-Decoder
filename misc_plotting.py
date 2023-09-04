import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal
import matplotlib.pyplot as plt
from maze_utils import Maze, RangeNormalize
from data_generation import data_to_index
from models import ClassifierMLP

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
        
        
        
def plot_spike_probs(clf, data, labels):
    norm = RangeNormalize()
    norm.fit(range_min = -50, range_max = 150)
    data = norm.transform(data)
    
    probs = clf.predict(data, return_log_probs = False).transpose(0,1)
    
    
    #plt.pcolormesh(probs)
    fig, ax = plt.subplots()
    ax.imshow(probs, aspect = 'auto', interpolation = 'gaussian',)
    ax.set_xticks(np.arange())


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
    
    
def plot_density_mixture_likelihood(
        mixture_model, input_data, label_data, posNorm, log_likelihood = False,
        ):
    latent = torch.linspace(0,1,100)
    
    X_likelihood = torch.zeros((input_data.size(0), 100))
    Y_likelihood = torch.zeros((input_data.size(0), 100))
    
    pi_k, mu_k, var_k = mixture_model(input_data)
    
    for i in range(input_data.size(0)):
        xtemp = torch.zeros_like(latent)
        ytemp = torch.zeros_like(latent)
        for k in range(mixture_model.num_mixtures):
            xtemp += torch.exp(
                Normal(mu_k[i,k,0], var_k[i,k,0]).log_prob(latent)
                ) * pi_k[i,k]
            ytemp += torch.exp(
                Normal(mu_k[i,k,1], var_k[i,k,1]).log_prob(latent)
                ) * pi_k[i,k]
        X_likelihood[i,:] = xtemp
        Y_likelihood[i,:] = ytemp
    
    if log_likelihood:
        X_likelihood = torch.log(X_likelihood).double()
        X_likelihood = torch.where(X_likelihood < -7.5, -7.5, X_likelihood)
        Y_likelihood = torch.log(Y_likelihood).double()
        Y_likelihood = torch.where(Y_likelihood < -7.5, -7.5, Y_likelihood)
        pi_k = torch.log(pi_k).double()
        pi_k = torch.where(pi_k < -7.5, -7.5, pi_k)
    
    fig, ax = plt.subplots(nrows = 3, sharex = True)
    fig.suptitle('Position Log-Likelihood Heatmap vs Time' if log_likelihood else 'Position Likelihood Heatmap vs Time')
    plt.xlabel('Time [s]')
    
    xmin, xmax = posNorm.range_min[0].item(), posNorm.range_max[0].item()
    ymin, ymax = posNorm.range_min[1].item(), posNorm.range_max[1].item()
    
    ax[0].pcolormesh(X_likelihood.T.detach(), cmap = 'viridis')
    ax[0].set_ylabel('X log-likelihood' if log_likelihood else 'X likelihood', fontsize = 8)
    xdelta = (xmax-xmin)/6
    ax[0].set_yticks([100/6,500/6])
    ax[0].set_yticklabels([int(xmin+xdelta), int(xmax-xdelta)], fontsize = 8)
    
    ax[1].pcolormesh(Y_likelihood.T.detach(), cmap = 'viridis')
    ax[1].set_ylabel('Y log-likelihood' if log_likelihood else 'Y likelihood', fontsize = 8)
    ydelta = (ymax-ymin)/6
    ax[1].set_yticks([100/6,500/6])
    ax[1].set_yticklabels([int(ymin+ydelta), int(ymax-ydelta)], fontsize = 8)
    
    ax[2].pcolormesh(pi_k.T.detach(), cmap = 'viridis')
    ax[2].set_ylabel('Mixture\nlog-likelihood' if log_likelihood else 'Mixture\nlikelihood', fontsize = 8)
    yticks = torch.arange(0.5, mixture_model.num_mixtures, 2)
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels([i.item() for i in (yticks+0.5).long()], fontsize = 8)
    
    xticks = torch.arange(0, label_data.size(0), int(20*30))
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels([(i/30).int().item() for i in xticks])


def plot_density_mixture_HPD(
        mixture_model, covariance_type, input_data, label_data, posNorm, alpha = 0.1,
        ):
    latent = torch.linspace(0,1,100)
    
    if covariance_type == 'full':
        latent = torch.cat(
            tuple(torch.dstack(torch.meshgrid(latent, latent, indexing = 'ij')))
            )
    
    X_likelihood = torch.ones((input_data.size(0), 100))
    Y_likelihood = torch.ones((input_data.size(0), 100))
    
    if covariance_type == 'diag':
        pi_k, mu_k, var_k = mixture_model(input_data)
    elif covariance_type == 'full':
        pi_k, mu_k, tril_k = mixture_model(input_data)
    
    xmin, xmax = posNorm.range_min[0].item(), posNorm.range_max[0].item()
    ymin, ymax = posNorm.range_min[1].item(), posNorm.range_max[1].item()
    
    res = 1.212
    ind = data_to_index(label_data, xmin, ymin, resolution = res, unique = False)
    
    score = 0
    area = 0
    
    if covariance_type == 'diag':
        
        for i in range(input_data.size(0)):
            xtemp = torch.zeros_like(latent)
            ytemp = torch.zeros_like(latent)
            
            for k in range(mixture_model.num_mixtures):
                xtemp += torch.exp(
                    Normal(mu_k[i,k,0], var_k[i,k,0]).log_prob(latent)
                    ) * pi_k[i,k]
                ytemp += torch.exp(
                    Normal(mu_k[i,k,1], var_k[i,k,1]).log_prob(latent)
                    ) * pi_k[i,k]
            
            xtemp /= xtemp.sum(0)
            ytemp /= ytemp.sum(0)
            
            xsort, xix = xtemp.sort(dim = 0, descending = True)
            ysort, yix = ytemp.sort(dim = 0, descending = True)
            
            xidx = 0
            while xsort[:xidx].sum(0) < (1 - alpha):
                xidx += 1
            X_likelihood[i,xix[:xidx]] = 0.4
            
            yidx = 0
            while ysort[:yidx].sum(0) < (1 - alpha):
                yidx += 1
            Y_likelihood[i,yix[:yidx]] = 0.4
            
            area += xidx * yidx
            
            xind = ind[i,0]
            yind = ind[i,1]
            if X_likelihood[i,xind] != 1 and Y_likelihood[i,yind] != 1:
                score += 1
            X_likelihood[i,xind-1:xind+2] = 0
            Y_likelihood[i,yind-1:yind+2] = 0
            
    elif covariance_type == 'full':
        
        for i in range(input_data.size(0)):
            temp = torch.zeros((latent.size(0),))
            
            for k in range(mixture_model.num_mixtures):
                temp += torch.exp(
                    MultivariateNormal(loc = mu_k[i,k,:], scale_tril = tril_k[i,k,:,:]).log_prob(latent)
                    ) * pi_k[i,k]
            
            sort, ix = temp.sort(dim = 0, descending = True)
            
            idx = 0
            while sort[:idx].sum(0) < (1 - alpha):
                idx += 1
            
            area += idx
            
            temp = torch.zeros_like(temp)
            temp[ix[:idx]] = 1
            temp = temp.view(100,100)
            
            xind, yind = ind[i,0], ind[i,1]
            
            if temp[xind,yind] == 1:
                score += 1
            
            xix = temp.sum(dim = 1).nonzero()
            yix = temp.sum(dim = 0).nonzero()
            
            X_likelihood[i,xix] = 0.4
            Y_likelihood[i,yix] = 0.4
            
            X_likelihood[i,xind-1:xind+2] = 0
            Y_likelihood[i,yind-1:yind+2] = 0
    
    score /= input_data.size(0)
    area /= (input_data.size(0) * 100**2)
    
    fig, ax = plt.subplots(nrows = 2, sharex = True)
    
    title = f'{int((1-alpha)*100)}'+'%'+' Highest Posterior Density Region (HDR) vs Time'
    title += '\nratio samples in HDR: %.4f' % score
    title += ' | mean HDR area coverage ratio: %.4f' % area
    fig.suptitle(title, fontsize = 10)
    plt.xlabel('Time [s]')
    
    ax[0].pcolormesh(X_likelihood.T.detach(), cmap = 'gist_stern')
    ax[0].set_ylabel('X Position', fontsize = 8)
    xdelta = (xmax-xmin)/6
    ax[0].set_yticks([100/6,500/6])
    ax[0].set_yticklabels([int(xmin+xdelta), int(xmax-xdelta)], fontsize = 8)
    
    ax[1].pcolormesh(Y_likelihood.T.detach(), cmap = 'gist_stern')
    ax[1].set_ylabel('Y Position', fontsize = 8)
    ydelta = (ymax-ymin)/6
    ax[1].set_yticks([100/6,500/6])
    ax[1].set_yticklabels([int(ymin+ydelta), int(ymax-ydelta)], fontsize = 8)
    
    xticks = torch.arange(0, label_data.size(0), int(20*30))
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels([(i/30).int().item() for i in xticks])
    
    return score, area
    
    
def plot_pretrained_learned_mixtures(maze, mixture_model, input_data, posNorm):
    
    pik, muk, vark = mixture_model(input_data)
    
    samp = posNorm.untransform(
        Normal(muk.mean(0), vark.mean(0)**0.5).sample((50,))
        )
    
    plt.figure()
    plt.plot(maze.pos[:,0], maze.pos[:,1], 'o', color = '0.4', label = 'maze', markersize = 0.5)
    for k in range(mixture_model.num_mixtures):
        plt.plot(samp[:,k,0], samp[:,k,1], 'o', label = f'mixture {k+1}', markersize = 2)
    plt.legend(fontsize = 8)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Learned Mixtures from Pre-trained Model')
    
    
    
    
# (9,3), (10,3)
if __name__ == '__main__':
    
    hl = 16
    bs = 1
    
    
    wm1 = Maze(
        name = 'Emile', 
        session = (20,6), 
        n_points = 'all', 
        )
# =============================================================================
#     wm2 = Maze(
#         name = 'Bon', 
#         session = (3,3), 
#         n_points = 'all', 
#         )
#         
# 
#     inputs1, labels1 = wm1.generate_observation_process_data(
#         history_length = hl, bin_size = bs, shuffle = False,
#         )
#     inputs2, labels2 = wm2.generate_observation_process_data(
#         history_length = hl, bin_size = bs, shuffle = False,
#         )
#     
#     inputs = torch.cat([inputs1, inputs2], dim = 0)
#     labels = torch.cat([labels1, labels2], dim = 0)
#     
#     b0 = int(inputs.size(0)*0.85)
#     b1 = int(inputs.size(0)*0.95)
#     
#     spikeNorm = RangeNormalize()
#     spikeNorm.fit(range_min = 0, range_max = 10)
#     
#     test_input = spikeNorm.transform(inputs[b1:]).flatten(1,-1)
#     test_label = labels[b1:]
# =============================================================================
    
    wm1.plot_position(plot_map = True)
    print(wm1.spike_counts.size())
    #plot_spikes_on_maze(wm1, single_plot = True)
    
    
# =============================================================================
#     clf = ClassifierMLP(
#         hidden_layer_sizes = [24,24], 
#         input_dim = test_input.size(1), 
#         num_classes = 5,
#         )
#     clf.load_state_dict(
#         torch.load('ObservationModels/HL16_BS1_2layerMLP_P_arm_given_input_hid24.pt'))
#         
#     plot_spike_probs(
#         clf = clf, data = test_input, labels = test_label,
#         )
# =============================================================================
    











