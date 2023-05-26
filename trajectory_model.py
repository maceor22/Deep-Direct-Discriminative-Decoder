from copy import deepcopy
import scipy.io as sio
from utils import *
from maze_utils import *
from state_process_models import *
from in_maze_model import InMazeModelNN, generate_inside_maze_prob_history
from data_generation import generate_trajectory_history
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time as time
           

# this needs to be updated; will not work with current models
def evaluate_history(sun_maze, max_history_length, 
                     plot_on = False, save_best_model = False):
    
    history_train_losses = []
    history_valid_losses = []
    history_test_losses = []
    
    history_lengths = range(1, max_history_length+1)
    
    best_test = 1e10
    
    for hl in history_lengths:
        
        print(f'\nhistory length: {hl}\n')
        
        data = sun_maze.generate_position_history(history_length = hl)[(max_history_length-hl):,:,:]
        
        input_data = data[:-1,:,:]
        label_data = data[1:,:,:2]
        
        bound1 = int(data.size(0)*.70)
        bound2 = int(data.size(0)*.90)
        
        train_input = input_data[:bound1,:,:]
        train_label = label_data[:bound1,-1,:]
        
        valid_input = input_data[bound1:bound2,:,:]
        valid_label = label_data[bound1:bound2,-1,:]
        
        test_input = input_data[bound2:,:,:]
        test_label = label_data[bound2:,-1,:]
        
        
        normalizer = NormalizeStateProcessInputs()
        
        train_input = normalizer.fit_transform(train_input)
        valid_input = normalizer.transform(valid_input)
        test_input  = normalizer.transform(test_input)
        
        train_label = normalizer.transform(train_label, normalize_vel = False)
        valid_label = normalizer.transform(valid_label, normalize_vel = False)
        test_label  = normalizer.transform(test_label, normalize_vel = False)
        
        
        train_data = Data(train_input, train_label)
        valid_data = Data(valid_input, valid_label)
        
        
        NN = StateProcessNN(
            hidden_layer_sizes = [32,32], 
            history_length = hl,
            )
        
        learning_rate = 1e-4
        
        optimizer = torch.optim.SGD(NN.parameters(), lr = learning_rate)
        
        best_epoch, train_losses, valid_losses = train(
            model = NN,
            train_data = train_data,
            valid_data = valid_data,
            optimizer = optimizer,
            epochs = 100,
            batch_size = 64,
            )
        
        history_train_losses.append(train_losses[best_epoch-1])
        history_valid_losses.append(valid_losses[best_epoch-1])
        
        test_mean, test_vars = NN.predict(test_input, return_sample = False)
        test_loss = nn.GaussianNLLLoss()(test_mean, test_label, test_vars)
        
        if test_loss < best_test:
            best_test = test_loss
            best_hl = hl
            best_state_dict = deepcopy(NN.state_dict())
        
        history_test_losses.append(test_loss.detach().numpy())
        
        print(f'\ntraining loss: {train_losses[best_epoch-1]} | validation loss: {valid_losses[best_epoch-1]} | test loss: {test_loss}')
        print('-------------------------------------------------------------------\n')
        
    
    print(f'\nBest history length: {best_hl} | Best test loss: {best_test}\n')
    
    
    if save_best_model:
        torch.save(best_state_dict, 'Models/best_hist_state_dict.pt')
    
        
    if plot_on:
        
        plt.figure()
        plt.plot(history_lengths, history_train_losses, '0.4', label = 'training')
        plt.plot(history_lengths, history_valid_losses, 'b', label = 'validation')
        plt.plot(history_lengths, history_test_losses, 'orange', label = 'testing')
        plt.xticks(history_lengths)
        plt.xlabel('History Length')
        plt.ylabel('Deviance [-2*LogLikelihood]')
        plt.title('Deviance versus History Length')
        plt.legend()



if __name__ == '__main__':
    
    # history length
    hl = 32
    
    # data transform
    tf = RangeNormalize()
    tf.fit(
        pos_mins = torch.Tensor([-50, -50]), pos_maxs = torch.Tensor([150, 150])
        )
    
    # load in-maze classifier
    insideMaze = InMazeModelNN(hidden_layer_sizes = [24,24], feature_dim = 2, history_length = hl)
    insideMaze.load_state_dict(torch.load(f'InMaze/insideMaze_AUC_2LayerMLP_state_dict_HL{hl}.pt'))
    
    # load in-maze data and generate in-maze probability history
    in_data = generate_inside_maze_prob_history(
        tf.transform(torch.load(f'Datasets/in_data_HL{hl}.pt')), insideMaze,
        )
    
    # split in-maze data into inputs and labels
    in_input, in_label = in_data[:-1,:,:], in_data[1:,-1,:-1]
    
    # load out-maze data and generate in-maze probability history
    out_data = generate_inside_maze_prob_history(
        data = tf.transform(generate_trajectory_history(
            data = torch.load('InMaze/out_trajectory.pt'), history_length = hl)),
        inside_maze_model = insideMaze, flatten = True,)
    
    # split out-maze data into inputs and labels
    out_input, out_label = out_data[:-1,:,:], out_data[1:,-1,:-1]
    
    # for balanced data between in-maze and out-maze data, randomly sample 
    #   points from out-maze data
    out_idxs = torch.randperm(out_input.size(0))[:in_input.size(0)]
    out_input, out_label = out_input[out_idxs], out_label[out_idxs]
        
    # set bounds used for creating train / validation / test datasets
    b0 = int(0.85*in_input.size(0))
    b1 = int(0.95*in_input.size(0))
    
    
    # aggregate in-maze and out-maze data into train / validation / test datasets
    
    train_input = torch.cat(
        [in_input[:b0], out_input[:b0]], dim = 0).flatten(1,-1)
    valid_input = torch.cat(
        [in_input[b0:b1], out_input[b0:b1]], dim = 0).flatten(1,-1)
    # interested in test performance on in-maze data
    test_input = in_input[b1:].flatten(1,-1)
    
    train_label = torch.cat(
        [in_label[:b0], out_label[:b0]], dim = 0)
    valid_label = torch.cat(
        [in_label[b0:b1], out_label[b0:b1]], dim = 0)
    # interested in test performance on in-maze data
    test_label = in_label[b1:]

    # train labels for modelA and modelB
    tr_labelA = train_label[:,0]
    tr_labelB = train_label[:,1]
    
    # validation labels for modelA and modelB
    va_labelA = valid_label[:,0]
    va_labelB = valid_label[:,1]
    
    # unnormalize test labels
    te_label = tf.untransform(test_label.unsqueeze(1)).squeeze()
    
    # initialize modelA
    P_x_given_input = PositionNN(hidden_layer_sizes = [32,32], input_dim = hl*3)
    #P_x_given_input.load_state_dict(torch.load(f'Models/P_x_y_HL{hl}/2layerMLP_P_x_given_input_hid32.pt'))
    
    # initialize modelB
    P_y_given_x_input = PositionNN(hidden_layer_sizes = [32,32], input_dim = hl*3+2)
    #P_y_given_x_input.load_state_dict(torch.load(f'Models/P_x_y_HL{hl}/2layerMLP_P_y_given_x_input_hid32.pt'))
    
    # initialize trainers for modelA and modelB
    trainerA = TrainerGaussNLL(
        optimizer = torch.optim.SGD(P_x_given_input.parameters(), lr = 1e-3))
    trainerB = TrainerGaussNLL(
        optimizer = torch.optim.SGD(P_y_given_x_input.parameters(), lr = 1e-3))
    
    # run training scheme
    TwoModelTrain(
        modelA = P_x_given_input, modelB = P_y_given_x_input, 
        trainerA = trainerA, trainerB = trainerB, 
        train_input = train_input, valid_input = valid_input, 
        train_labelA = tr_labelA, valid_labelA = va_labelA, 
        train_labelB = tr_labelB, valid_labelB = va_labelB,
        epochs = (300,400), batch_size = 256, plot_losses = True,
        )
    
    # save models
    torch.save(P_x_given_input.state_dict(), f'Models/P_x_y_HL{hl}/2layerMLP_P_x_given_input_hid32.pt')
    torch.save(P_y_given_x_input.state_dict(), f'Models/P_x_y_HL{hl}/2layerMLP_P_y_given_x_input_hid32.pt')
    
    # load modelA and modelB into two-model wrapper class
    JointProbXY = TwoModelWrapper(
        modelA = P_x_given_input, modelB = P_y_given_x_input, transform = tf,
        )
    
    # evaluate performance (MSE) on test data
    pred = JointProbXY.predict(
        test_input, return_sample = True, untransform_data = True,)
    
    pred_MSE = nn.MSELoss()(pred, te_label)
    print('\ntest mse: %.3f' % pred_MSE)
    
    # plot performance on test data
    pred_mean, pred_vars = JointProbXY.predict(
        test_input, return_sample = False, untransform_data = True,)
    
    plot_model_predictions(
        pred_mean, pred_vars, te_label, 
        title = f'P(X,Y|input) = P(Y|X,input) * P(X|input) | HL = {hl}\nTest Predictions (MSE: %.3f)' % pred_MSE,
        )





