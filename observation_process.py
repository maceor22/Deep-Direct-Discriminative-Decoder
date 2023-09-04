import os
from copy import deepcopy
import torch
import numpy as np
from torch import nn
from torch import autograd as ag
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from misc_plotting import *
from maze_utils import Maze, Data, plot_model_predictions, RangeNormalize, WMazeDiscreteTransform2
from models import *
from trainers import *
from data_generation import data_to_index
import time as time



def evaluate_history_arm_distance(mazes, max_history, min_history = 0, plot = False):
    
    spikeNorm = RangeNormalize()
    spikeNorm.fit(range_min = 0, range_max = 100)
    
    train_MSEs = []
    test_MSEs = []
    
    for hl in range(min_history, max_history+1):
        print(f'------------ HL: {hl} ------------')
        inputs = []
        labels = []
        for maze in mazes:
            new_input, new_label = maze.generate_observation_process_data(
                history_length = hl, bin_size = 1, shuffle = False,
                )
            inputs.append(new_input)
            labels.append(new_label)
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        b0 = int(inputs.size(0)*0.85)
        b1 = int(inputs.size(0)*0.95)        

        posNorm = WMazeDiscreteTransform2()
        
        train_input = spikeNorm.transform(inputs[:b0]).flatten(1,-1)
        train_label = posNorm.fit_transform(labels[:b0])
        
        valid_input = spikeNorm.transform(inputs[b0:b1]).flatten(1,-1)
        valid_label = posNorm.transform(labels[b0:b1])
        
        test_input = spikeNorm.transform(inputs[b1:]).flatten(1,-1)
        test_label = labels[b1:]
        
        tr_labelA = train_label[:,:-1].argmax(dim = 1)
        tr_labelB = train_label[:,-1]
        
        va_labelA = valid_label[:,:-1].argmax(dim = 1)
        va_labelB = valid_label[:,-1]
        
        P_arm_given_input = ClassifierMLP(
            hidden_layer_sizes = [64,64], 
            input_dim = inputs.size(1)*inputs.size(2),
            num_classes = 5,
            )
        
        P_dist_given_arm_input = GaussianMLP(
            hidden_layer_sizes = [64,64], 
            input_dim = inputs.size(1)*inputs.size(2)+5,
            latent_dim = 1,
            )
        
        trainerA = TrainerNLL(
            optimizer = torch.optim.SGD(P_arm_given_input.parameters(), lr = 1e-3),
            suppress_prints = True,
            )
        trainerB = TrainerGaussNLL(
            optimizer = torch.optim.SGD(P_dist_given_arm_input.parameters(), lr = 1e-3),
            suppress_prints = True,
            )

        # run training scheme
        TwoModelTrain(
            modelA = P_arm_given_input, modelB = P_dist_given_arm_input, 
            trainerA = trainerA, trainerB = trainerB, 
            train_input = train_input, valid_input = valid_input, 
            train_labelA = tr_labelA, valid_labelA = va_labelA, 
            train_labelB = tr_labelB, valid_labelB = va_labelB,
            modelA_multi_output = False,
            epochs = (100,100), batch_size = 256, plot_losses = False,
            )
        
        # save models
        torch.save(P_arm_given_input.state_dict(), f'ObservationModels/HL{hl}_BS1_2layerMLP_hid24_P_arm_given_input.pt')
        torch.save(P_dist_given_arm_input.state_dict(), f'ObservationModels/HL{hl}_BS1_2layerMLP_hid24_P_dist_given_arm_input.pt')
        
        # load modelA and modelB into two-model wrapper class
        JointProbXY = TwoModelWrapper(
            modelA = P_arm_given_input, modelB = P_dist_given_arm_input, 
            transform = posNorm, modelA_multi_output = False,
            )
        
        # evaluate performance (MSE) on train and test data
        train_pred = JointProbXY.predict(
            train_input, return_sample = True, untransform_data = True,)
        test_pred = JointProbXY.predict(
            test_input, return_sample = True, untransform_data = True,)
        
        train_MSE = nn.MSELoss()(train_pred, labels[:b0])
        test_MSE = nn.MSELoss()(test_pred, test_label)
        train_MSEs.append(train_MSE)
        test_MSEs.append(test_MSE)
        print('train MSE: %.3f' % train_MSE)
        print('test MSE: %.3f' % test_MSE)
        
    if plot:
        plt.figure()
        plt.plot(range(min_history, max_history+1), train_MSEs, '0.4', label = 'training')
        plt.plot(range(min_history, max_history+1), test_MSEs, 'blue', label = 'testing')
        plt.legend()
        plt.xlabel('History Length')
        plt.ylabel('Mean Square Error')
        plt.title('Effect of History Length on MSE')
        

def train_one_model(
        model, trainer, 
        inputs, labels, 
        input_norm, label_norm, 
        bounds, epochs = 100, batch_size = 256,
        ):
    b0, b1 = bounds[0], bounds[1]
    
    bal_train_input, bal_train_label = balance_dataset(
        input_data = inputs[:b0], label_data = labels[:b0], 
        xmin = -10, xmax = 110, ymin = -10, ymax = 110, 
        resolution = 1.0, threshold = 50,
        )
    bal_valid_input, bal_valid_label = balance_dataset(
        input_data = inputs[b0:b1], label_data = labels[b0:b1], 
        xmin = -10, xmax = 110, ymin = -10, ymax = 110, 
        resolution = 1.0, threshold = 50,
        )
    
    train_input = input_norm.transform(bal_train_input).flatten(1,-1)
    valid_input = input_norm.transform(bal_valid_input).flatten(1,-1)
    test_input = input_norm.transform(inputs[b1:]).flatten(1,-1)
    
    train_label = label_norm.transform(bal_train_label)
    valid_label = label_norm.transform(bal_valid_label)
    test_label = labels[b1:]
    
    trainer.train(
        model = model, 
        train_data = Data(train_input, train_label), 
        valid_data = Data(valid_input, valid_label),
        epochs = epochs, batch_size = batch_size, plot_losses = False,
        )
    
    return


def evaluate_history_XY(mazes, num_folds, test_size, max_history, min_history = 0, plot = False):
    
    spikeNorm = RangeNormalize()
    spikeNorm.fit(range_min = 0, range_max = 10)
    
    posNorm = RangeNormalize()
    posNorm.fit(range_min = (110, 10), range_max = (310, 210))
    
    train_MSE = torch.zeros((max_history-min_history+1,num_folds))
    test_MSE = torch.zeros((max_history-min_history+1,num_folds))
    
    for hl in range(min_history, max_history+1):
        print(f'------------ HL: {hl} ------------')
        inputs = []
        labels = []
        for maze in mazes:
            new_input, new_label = maze.generate_observation_process_data(
                history_length = hl, bin_size = 1, shuffle = False,
                )
            inputs.append(new_input)
            labels.append(new_label)
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)
        
        ix = torch.arange(inputs.size(0))
        
        upper = inputs.size(0)
        b1 = upper - test_size
        b0 = b1 - 2*test_size
        lower = test_size*(num_folds-1)
        
        for k in range(num_folds):
            train_ix = ix[lower:b0]
            valid_ix = ix[b0:b1]
            test_ix = ix[b1:upper]
            
            bal_train_input, bal_train_label = balance_dataset(
                input_data = inputs[train_ix], label_data = labels[train_ix], 
                xmin = 150, xmax = 270, ymin = 50, ymax = 170, 
                resolution = 1.0, threshold = 50,
                )
            bal_valid_input, bal_valid_label = balance_dataset(
                input_data = inputs[valid_ix], label_data = labels[valid_ix], 
                xmin = 150, xmax = 270, ymin = 50, ymax = 170, 
                resolution = 1.0, threshold = 50,
                )
            
            train_input = spikeNorm.transform(bal_train_input).flatten(1,-1)
            valid_input = spikeNorm.transform(bal_valid_input).flatten(1,-1)
            
            train_label = posNorm.transform(bal_train_label)
            valid_label = posNorm.transform(bal_valid_label)
            
            test_input = spikeNorm.transform(inputs[test_ix]).flatten(1,-1)
            test_label = labels[test_ix]
            
            JointProbXY = GaussianMLP(
                hidden_layer_sizes = [24,24], 
                input_dim = inputs.size(1)*inputs.size(2),
                latent_dim = 2,
                )
            
            trainer = TrainerGaussNLL(
                optimizer = torch.optim.SGD(JointProbXY.parameters(), lr = 1e-3),
                suppress_prints = True,
                )
        
            trainer.train(
                model = JointProbXY, 
                train_data = Data(train_input, train_label), 
                valid_data = Data(valid_input, valid_label),
                epochs = 100, batch_size = 256, plot_losses = False,
                )
            
            #torch.save(JointProbXY.state_dict(), f'ObservationModels/HL{hl}_BS{bs}_2layerMLP_JointProbXY_hid24.pt')
            
            train_pred = posNorm.untransform(
                JointProbXY.predict(train_input, return_sample = True)
                )
            test_pred = posNorm.untransform(
                JointProbXY.predict(test_input, return_sample = True)
                )
            
            tr_MSE = nn.MSELoss()(train_pred, bal_train_label).item()
            te_MSE = nn.MSELoss()(test_pred, test_label).item()
            
            train_MSE[hl-min_history,k] = tr_MSE
            test_MSE[hl-min_history,k] = te_MSE
            
            print(f'fold {k+1} completed')
            
            lower -= test_size
            b0 -= test_size
            b1 -= test_size
            upper -= test_size
            
        
    if plot:
        tr_mse_mean = train_MSE.mean(dim = 1)
        te_mse_mean = test_MSE.mean(dim = 1)
        
        tr_mse_std = train_MSE.std(dim = 1)
        te_mse_std = test_MSE.std(dim = 1)
        
        haxis = range(min_history, max_history+1)
        plt.figure()
        plt.plot(haxis, tr_mse_mean, '0.2', label = 'train mean')
        plt.fill_between(
            haxis, tr_mse_mean + tr_mse_std, tr_mse_mean - tr_mse_std,
            color = '0.6', label = 'train sdev', alpha = 0.8,
            )
        plt.plot(haxis, te_mse_mean, 'navy', label = 'test mean')
        plt.fill_between(
            haxis, te_mse_mean + te_mse_std, te_mse_mean - te_mse_std,
            color = 'lightskyblue', label = 'test sdev', alpha = 0.4,
            )
        plt.xticks(np.arange(min_history, max_history+1, step = (max_history-min_history)/10))
        plt.legend(fontsize = 8)
        plt.xlabel('History Length')
        plt.ylabel('Mean Square Error')
        plt.title(f'Effect of History Length on MSE\n{num_folds}-Fold Cross Validation')
        


def balance_dataset(
        input_data, label_data, 
        xmin, xmax, ymin, ymax, resolution,
        threshold = None, presence = False, p_range = 1, 
        delta_included = False, dmin = None, dmax = None,
        plotting = False,
        ):
    
    label_data = label_data.unsqueeze(1) if label_data.dim() == 2 else label_data
    
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
                sample_size = (density*grid_tensor[i,j]).ceil().long()
            else:
                density = grid_tensor[i,j].long()
                data_ix = density-1
                sample_size = min(len(grid_list[i][j]), threshold)
            
            if density > 0:
                if density <= threshold: data[data_ix] += 1
                ixs = torch.tensor(grid_list[i][j])
                ixs = ixs[torch.randperm(ixs.size(0))][:sample_size]
                indices += [ix.item() for ix in ixs]
    
    indices = torch.tensor(indices).long()
    if delta_included:
        if dmin == None:
            dmin = -1
        if dmax == None:
            dmax = 1e3
        ixs = []
        for ind in indices:
            l_inf_norm = label_data[ind,:,2:].squeeze().abs().max(0)[0]
            if l_inf_norm > dmin and l_inf_norm < dmax:
                ixs.append(ind.unsqueeze(0))
        indices = torch.cat(ixs, dim = 0)
    
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
        
    return input_data[indices], label_data[indices].squeeze()
        


def get_position_delta(position_data, return_mean = False):
    for i in range(position_data.size(1)-1):
        position_data[:,i,:] -= position_data[:,-1,:]
    if return_mean:
        return torch.cat([
            position_data[:,-1,:].unsqueeze(1),
            position_data[:,:-1,:].mean(1).unsqueeze(1),
            ], dim = 1)
    else:
        return position_data



def generate_dataset(
        rat_name, spike_history_length, spike_bin_size, position_history_length,
        include_velocity = False, dirvel = True, 
        resolution = 10, threshold = None, 
        presence = False, p_range = 1, 
        delta_included = False, dmin = None, dmax = None,
        ):
    
    if rat_name == 'Bon':
        xmin, xmax, ymin, ymax = 150, 270, 50, 170
    elif rat_name == 'Emile':
        xmin, xmax, ymin, ymax = -10, 260, -10, 210
    
    path = 'ObservationModels/datasets/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        rat_name, spike_history_length, spike_bin_size, position_history_length,
        include_velocity, dirvel, resolution, threshold, presence, p_range,
        delta_included, dmin, dmax,
        )
    
    if os.path.exists(path):
        print('loading datasets.....')
        bal_train_input = torch.load(path + '/train_input.pt')
        bal_train_label = torch.load(path + '/train_label.pt')
        bal_valid_input = torch.load(path + '/valid_input.pt')
        bal_valid_label = torch.load(path + '/valid_label.pt')
        raw_test_input = torch.load(path + '/test_input.pt')
        test_label = torch.load(path + '/test_label.pt')
                
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
        
        
        elif rat == 'Emile':
            
            wm1 = Maze(
                name = 'Emile', 
                session = (20,1), 
                n_points = 'all',
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm2 = Maze(
                name = 'Emile', 
                session = (20,3), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm3 = Maze(
                name = 'Emile', 
                session = (20,4), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            wm4 = Maze(
                name = 'Emile', 
                session = (20,6), 
                n_points = 'all', 
                include_velocity = include_velocity,
                dirvel = dirvel,
                )
            mazes = (wm1, wm2, wm3, wm4)
            
        
        print('\ngenerating history terms.....')
        inputs, labels = [], []
        for m in mazes:
            new_inputs, new_labels = m.generate_observation_process_data(
                spike_history_length = spike_history_length, spike_bin_size = spike_bin_size, 
                position_history_length = position_history_length, shuffle = False,
                )
            if phl > 0 and not include_velocity:
                new_labels = get_position_delta(new_labels, return_mean = True).flatten(1,-1)
            inputs.append(new_inputs)
            labels.append(new_labels)
        
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)
        
        
        b0 = int(inputs.size(0)*0.85)
        b1 = int(inputs.size(0)*0.95)
        
        
        print('\nbalancing train & validation data.....')
        bal_train_input, bal_train_label = balance_dataset(
            input_data = inputs[:b0], label_data = labels[:b0], 
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = resolution, threshold = threshold, presence = presence, p_range = p_range, 
            delta_included = include_velocity, dmin = dmin, dmax = dmax, plotting = False,
            )
        bal_valid_input, bal_valid_label = balance_dataset(
            input_data = inputs[b0:b1], label_data = labels[b0:b1], 
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, 
            resolution = resolution, threshold = threshold, presence = presence, p_range = p_range, 
            delta_included = include_velocity, dmin = dmin, dmax = dmax, plotting = False,
            )
        
        raw_test_input = inputs[b1:]
        test_label = labels[b1:,:2]
        
        os.mkdir(path)
        
        torch.save(bal_train_input, path + '/train_input.pt')
        torch.save(bal_train_label, path + '/train_label.pt')
        torch.save(bal_valid_input, path + '/valid_input.pt')
        torch.save(bal_valid_label, path + '/valid_label.pt')
        torch.save(raw_test_input, path + '/test_input.pt')
        torch.save(test_label, path + '/test_label.pt')
    
    ret = {}
    ret['train_input'] = bal_train_input
    ret['train_label'] = bal_train_label
    ret['valid_input'] = bal_valid_input
    ret['valid_label'] = bal_valid_label
    ret['test_input'] = raw_test_input
    ret['test_label'] = test_label
    ret['xmin'], ret['xmax'], ret['ymin'], ret['ymax'] = xmin, xmax, ymin, ymax
    
    return ret


def pretrain_density_mixture_network(
        mixture_network, covariance_type, 
        train_input, train_label, valid_input, valid_label,
        epochs = 100, batch_size = 256, plot_losses = False,
        ):
    gmm = GaussianMixture(
        n_components = mixture_network.num_mixtures, 
        covariance_type = covariance_type,
        tol = 1e-9, max_iter = 1000,
        )
    
    train_z = torch.from_numpy(gmm.fit_predict(train_label))
    valid_z = torch.from_numpy(gmm.predict(valid_label))
    
    mu = torch.from_numpy(gmm.means_)
    var = torch.from_numpy(gmm.covariances_)
    
    tr_pi_target = torch.zeros((train_z.size(0), mixture_network.num_mixtures))
    tr_pi_target[torch.arange(train_z.size(0)),train_z] = 1
    tr_pi_target = nn.LogSoftmax(dim = 1)(
        tr_pi_target + torch.zeros_like(tr_pi_target).uniform_(1e-3,0.05)
        )
    
    tr_mu_target = torch.zeros((train_z.size(0),mu.size(0), mu.size(1))).uniform_()
    
    if covariance_type == 'diag':
        tr_var_target = torch.zeros_like(tr_mu_target).uniform_(0.05,0.1)
    elif covariance_type == 'full':
        tr_var_target = torch.zeros_like(var).repeat(train_z.size(0),1,1,1)
        tr_var_target[:,:,0,0] = torch.rand((train_z.size(0),var.size(0))) * 0.05 + 0.05
        tr_var_target[:,:,1,1] = torch.rand((train_z.size(0),var.size(0))) * 0.05 + 0.05
    
    for i in range(train_z.size(0)):
        k = train_z[i]
        tr_mu_target[i,k] = mu[k]
        tr_var_target[i,k] = var[k]
        
    va_pi_target = torch.zeros((valid_z.size(0), mixture_network.num_mixtures))
    va_pi_target[torch.arange(valid_z.size(0)),valid_z] = 1
    va_pi_target = nn.LogSoftmax(dim = 1)(
        va_pi_target + torch.zeros_like(va_pi_target).uniform_(1e-3,0.05)
        )
    
    va_mu_target = torch.zeros((valid_z.size(0),mu.size(0), mu.size(1))).uniform_()
    
    if covariance_type == 'diag':
        va_var_target = torch.zeros_like(va_mu_target).uniform_(0.05,0.1)
    elif covariance_type == 'full':
        va_var_target = torch.zeros_like(var).repeat(valid_z.size(0),1,1,1)
        va_var_target[:,:,0,0] = torch.rand((valid_z.size(0),var.size(0))) * 0.05 + 0.05
        va_var_target[:,:,1,1] = torch.rand((valid_z.size(0),var.size(0))) * 0.05 + 0.05
        
    for i in range(valid_z.size(0)):
        k = valid_z[i]
        va_mu_target[i,k] = mu[k]
        va_var_target[i,k] = var[k]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixture_network = mixture_network.to(device)
    
    train_losses = []
    valid_losses = []
    
    stime = time.time()
    
    best_loss = 1e10
    
    optimizer = torch.optim.SGD(mixture_network.parameters(), lr = 1e-3)
    
    for epoch in range(1,epochs+1):        
        train_loss = 0
        valid_loss = 0
        
        ix = torch.randperm(train_z.size(0))
        lower = 0
        upper = batch_size
        
        mixture_network.train()
        # iterate over the training data
        while upper <= train_z.size(0):
            input_batch = ag.Variable(train_input[ix[lower:upper]].to(device))
            
            #pi_target = ag.Variable(train_z[ix[lower:upper]].to(device))
            pi_target = ag.Variable(tr_pi_target[ix[lower:upper]].to(device))
            mu_target = ag.Variable(tr_mu_target[ix[lower:upper]].to(device))
            var_target = ag.Variable(tr_var_target[ix[lower:upper]].to(device))
            
            if covariance_type == 'diag':
                samp = Normal(mu_target, var_target**0.5).sample()
                
                pik, muk, vark = mixture_network(input_batch)
                
            elif covariance_type == 'full':
                samp = MultivariateNormal(mu_target, var_target.float()).sample()
                
                pik, muk, trilk = mixture_network(input_batch)
                vark = trilk @ trilk.transpose(2,3)
            
            loss = nn.KLDivLoss(
                reduction = 'batchmean', log_target = True)(torch.log(pik), pi_target)
            
            if covariance_type == 'diag':    
                loss = loss + nn.GaussianNLLLoss()(muk, samp, vark)
            elif covariance_type == 'full':
                loss = loss - MultivariateNormal(muk, vark).log_prob(samp).mean(0).mean(0)
            loss = loss / 2
            
            # zero gradients
            optimizer.zero_grad()
            # backpropagate loss
            loss.backward()
            # prevent exploding gradients
            clip_grad_value_(mixture_network.parameters(), clip_value = 2)
            # update weights
            optimizer.step()
            # aggregate training loss
            train_loss += loss.item()
            
            lower += batch_size
            upper += batch_size
        
        # compute mean training loss and save to list
        train_loss /= train_z.size(0) // batch_size
        train_losses.append(train_loss)
        
        mixture_network.eval()
        with torch.no_grad():
            ix = torch.randperm(valid_z.size(0))
            
            lower = 0
            upper = batch_size
            
            # iterate over validation data
            while upper <= valid_z.size(0):
                input_batch = valid_input[ix[lower:upper]]
                
                #pi_target = valid_z[ix[lower:upper]]
                pi_target = tr_pi_target[ix[lower:upper]]
                mu_target = va_mu_target[ix[lower:upper]]
                var_target = va_var_target[ix[lower:upper]]
                
                if covariance_type == 'diag':
                    samp = Normal(mu_target, var_target**0.5).sample()
                    
                    pik, muk, vark = mixture_network(input_batch)
                    
                elif covariance_type == 'full':
                    samp = MultivariateNormal(mu_target, var_target.float()).sample()
                    
                    pik, muk, trilk = mixture_network(input_batch)
                    vark = trilk @ trilk.transpose(2,3)
                    
                loss = nn.KLDivLoss(
                    reduction = 'batchmean', log_target = True)(torch.log(pik), pi_target)
                
                if covariance_type == 'diag':
                    loss = loss + nn.GaussianNLLLoss()(muk, samp, vark)
                elif covariance_type == 'full':
                    loss = loss - MultivariateNormal(muk, vark).log_prob(samp).mean(0).mean(0)
                    
                loss = loss / 2
                # compute and aggregate validation loss 
                valid_loss += loss.item()
                
                lower += batch_size
                upper += batch_size
        
        # compute mean validation loss and save to list
        valid_loss /= valid_z.size(0) // batch_size
        valid_losses.append(valid_loss)
        
        # save model that performs best on validation data
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state_dict = deepcopy(mixture_network.state_dict())
        
        # printing
        if epoch % 10 == 0:
            print(f'------------------{epoch}------------------')
            print('training loss: %.2f | validation loss: %.2f' % (
                train_loss, valid_loss))
            
            time_elapsed = (time.time() - stime)
            pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
            
            print('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                time_elapsed, pred_time_remaining))
    
    # load best model
    mixture_network.load_state_dict(best_state_dict)
    
    # plot loss curves
    if plot_losses:
        
        Train_losses = np.array(train_losses)
        Train_losses[np.where(Train_losses > 100)] = 100
        
        Valid_losses = np.array(valid_losses)
        Valid_losses[np.where(Valid_losses > 100)] = 100
        
        plt.figure()
        
        plt.plot(range(1,epochs+1), Train_losses, '0.4', label = 'training')
        plt.plot(range(1,epochs+1), Valid_losses, 'b', label = 'validation')
        
        plt.xlabel('Epochs')
        plt.xticks(range(0,epochs+1,int(epochs // 10)))
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()
        
    return best_epoch, train_losses, valid_losses
    



if __name__ == '__main__':
    
    shl = 20
    bs = 1
    
    data = generate_dataset(
        rat_name = 'Bon', 
        spike_history_length = shl, spike_bin_size = bs, position_history_length = 0, 
        include_velocity = True, dirvel = False, dmin = 0.5, dmax = 20,
        resolution = 0.1, threshold = 100, presence = False, p_range = 2,
        )
    
    bal_train_input, bal_train_label = data['train_input'], data['train_label']
    bal_valid_input, bal_valid_label = data['valid_input'], data['valid_label']
    
    raw_test_input, test_label = data['test_input'], data['test_label']
    
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    
    print('\n ', bal_train_input.size(), bal_train_label.size())
    
    spikeNorm = RangeNormalize(dim = bal_train_input.size(2), norm_mode = 'auto')
    spikeNorm.fit(
        range_min = [0 for _ in range(bal_train_input.size(2))], 
        range_max = [10 for _ in range(bal_train_input.size(2))],
        )
    
    train_input = spikeNorm.transform(bal_train_input).flatten(1,-1)
    valid_input = spikeNorm.transform(bal_valid_input).flatten(1,-1)
    test_input = spikeNorm.transform(raw_test_input).flatten(1,-1)
    
    two_model = False
    
    if two_model:
        
        posNorm = RangeNormalize(dim = 4, norm_mode = (0,0,1,1))
        posNorm.fit(
            range_min = (xmin, ymin, -20, -20), range_max = (xmax, ymax, 20, 20),
            )
        
        train_label = posNorm.transform(bal_train_label)
        valid_label = posNorm.transform(bal_valid_label)
        
        # train labels for modelA and modelB
        tr_labelA = train_label[:,2:]
        tr_labelB = train_label[:,:2]
        
        # validation labels for modelA and modelB
        va_labelA = valid_label[:,2:]
        va_labelB = valid_label[:,:2]
        
        
        P_dXdY_given_input = MultivariateNormalMLP(
            hidden_layer_sizes = [24,24], 
            input_dim = train_input.size(1),
            latent_dim = 2,
            )
        
        P_XY_given_dXdY_input = MultivariateNormalMLP(
            hidden_layer_sizes = [24,24], 
            input_dim = train_input.size(1)+6,
            latent_dim = 2,
            )
        
# =============================================================================
#         P_dXdY_given_input = GaussianMixtureMLP(
#             hidden_layer_sizes = [24, 24], 
#             num_mixtures = 2, 
#             input_dim = train_input.size(1), 
#             latent_dim = 2,
#             )
#         
#         P_XY_given_dXdY_input = GaussianMixtureMLP(
#             hidden_layer_sizes = [24, 24], 
#             num_mixtures = 2, 
#             input_dim = train_input.size(1)+4, 
#             latent_dim = 2,
#             )
# =============================================================================

        # initialize trainers for modelA and modelB
        trainerA = TrainerMultivariateNormalNLL(
            optimizer = torch.optim.SGD(P_dXdY_given_input.parameters(), lr = 1e-3))
        trainerB = TrainerMultivariateNormalNLL(
            optimizer = torch.optim.SGD(P_XY_given_dXdY_input.parameters(), lr = 1e-3))

# =============================================================================
#         trainerA = TrainerExpectationMaximization(
#             optimizer = torch.optim.SGD(P_dXdY_given_input.parameters(), lr = 1e-3))
#         trainerB = TrainerExpectationMaximization(
#             optimizer = torch.optim.SGD(P_XY_given_dXdY_input.parameters(), lr = 1e-3))
# =============================================================================

        # run training scheme
        TwoModelTrain(
            modelA = P_dXdY_given_input, modelB = P_XY_given_dXdY_input, 
            trainerA = trainerA, trainerB = trainerB, 
            train_input = train_input, valid_input = valid_input, 
            train_labelA = tr_labelA, valid_labelA = va_labelA, 
            train_labelB = tr_labelB, valid_labelB = va_labelB,
            modelA_multi_output = True,
            epochs = (1000,1000), batch_size = 256, plot_losses = True,
            )
        
        # save models
        torch.save(
            P_dXdY_given_input.state_dict(), 
            f'ObservationModels/models/HL{shl}_BS{bs}_2layerMLP_P_dXdY_given_input_hid24.pt')
        torch.save(
            P_XY_given_dXdY_input.state_dict(), 
            f'ObservationModels/HL{shl}_BS{bs}_2layerMLP_P_XY_given_dXdY_input_hid24.pt')
        
        # load modelA and modelB into two-model wrapper class
        JointProbXY = TwoModelWrapper(
            modelA = P_dXdY_given_input, modelB = P_XY_given_dXdY_input, 
            transform = None, modelA_multi_output = True,
            )
        
        # evaluate performance (MSE) on test data
        pred = JointProbXY.predict(test_input, return_sample = True)
        temp = pred[:,:2]
        pred[:,:2] = pred[:,2:]
        pred[:,2:] = temp
        pred = posNorm.untransform(pred)
        
        pred_MSE = nn.MSELoss()(pred[:,:2], test_label)
        print('\ntest mse: %.3f' % pred_MSE)
        
        # plot performance on test data
        pred_mean, pred_vars = JointProbXY.predict(
            test_input, return_sample = False,
            )
        temp1 = pred_mean[:,:2]
        pred_mean[:,:2] = pred_mean[:,2:]
        pred_mean[:,2:] = temp1
        pred_mean, pred_vars = posNorm.untransform(
            pred_mean, variance = torch.cat([
                pred_vars[:,2,0].view(-1,1), pred_vars[:,3,1].view(-1,1),
                pred_vars[:,0,0].view(-1,1), pred_vars[:,1,1].view(-1,1),
                ], dim = 1)
            )
        
        plot_model_predictions(
            pred_mean[:,:2], pred_vars[:,:2], test_label, 
            title = f'P(XY,dXdY|input) = P(XY|dXdY,input) * P(dXdY|input) | HL = {shl}, BS = {bs}\nTest Predictions (MSE: %.3f)' % pred_MSE,
            )
        
    else:
        
        posNorm = RangeNormalize(dim = 2, norm_mode = [0,0,])
        posNorm.fit(
            range_min = (xmin, ymin,), range_max = (xmax, ymax,),
            )
        
        train_label = posNorm.transform(bal_train_label[:,:2]).float()
        valid_label = posNorm.transform(bal_valid_label[:,:2]).float()

        print(train_input.size(), valid_input.size(), test_input.size())
        
# =============================================================================
#         JointProbXY = GaussianMixtureMLP(
#             hidden_layer_sizes = [24, 24], 
#             num_mixtures = 2, 
#             input_dim = train_input.size(1), 
#             latent_dim = 2,
#             )
#         
#         JointProbXY.load_state_dict(torch.load(
#             f'ObservationModels/pretrained/HL{shl}_BS{bs}_2layerGMMLP_{JointProbXY.num_mixtures}mix_JointProbXY_hid24.pt'))
# =============================================================================

        JointProbXY = MultivariateNormalMixtureMLP(
            hidden_layer_sizes = [24, 24], 
            num_mixtures = 5, 
            input_dim = train_input.size(1), 
            latent_dim = 2,
            )
        
# =============================================================================
#         JointProbXY.load_state_dict(torch.load(
#             f'ObservationModels/pretrained/HL{shl}_BS{bs}_2layerMVNMMLP_{JointProbXY.num_mixtures}mix_JointProbXY_hid24.pt'))
# =============================================================================
        
        
        print('\npre-training density mixture network.....\n')
        pretrain_density_mixture_network(
            mixture_network = JointProbXY, 
            covariance_type = 'full', 
            train_input = train_input, train_label = train_label, 
            valid_input = valid_input, valid_label = valid_label,
            epochs = 2000, plot_losses = True,
            )
        torch.save(JointProbXY.state_dict(),
                   f'ObservationModels/pretrained/HL{shl}_BS{bs}_2layerMVNMMLP_{JointProbXY.num_mixtures}mix_JointProbXY_hid24.pt')
        

# =============================================================================
#         trainer = TrainerExpectationMaximization(
#             optimizer = torch.optim.SGD(JointProbXY.parameters(), lr = 1e-3))
#         
#         trainer.train(
#             model = JointProbXY, 
#             train_data = Data(train_input, train_label), 
#             valid_data = Data(valid_input, valid_label),
#             epochs = 3000, batch_size = 256, plot_losses = True,
#             )
#         
#         torch.save(JointProbXY.state_dict(), 
#                    f'ObservationModels/models/HL{shl}_BS{bs}_2layerGMMLP_{JointProbXY.num_mixtures}mix_JointProbXY_hid24.pt')
# =============================================================================
    

# =============================================================================
#         score, area = plot_density_mixture_HPD(
#             mixture_model = JointProbXY, 
#             input_data = test_input, label_data = test_label, 
#             posNorm = posNorm, alpha = 0.05,
#             )    
# =============================================================================
        
    
    
    
    




