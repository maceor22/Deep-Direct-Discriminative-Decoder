import os
from copy import deepcopy
from maze_utils import *
from misc_plotting import *
from in_maze_model import GridClassifier, generate_inside_maze_prob_history
from data_generation import generate_dataset, generate_trajectory_history
from models import *
from trainers import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.distributions import Normal
import datetime
           

class StateProcess(nn.Module):

    def __init__(
            self, maze_classifier, transition_model, 
            history_length, mode = 1,
        ):

        super(StateProcess, self).__init__()

        self.clf = maze_classifier
        self.transition = transition_model
        self.history_length = history_length
        self.mode = mode

    def resample(self, xi):
        done = False
        resample_steps = 0
        while not done:
            samp = self.transition.predict(xi)
            if self.clf(samp) == self.mode:
                done = True
            resample_steps += 1

            if resample_steps % 100 == 0:
                done = True

        return samp, resample_steps

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        maze_hist = torch.stack(
            [self.clf(x[i]).unsqueeze(1) for i in range(x.size(0))], dim = 0)
        x = torch.cat([x, maze_hist], dim = -1)
        
        if return_sample:
            samples = self.transition.predict(x)
            return samples


class StateProcess1(nn.Module):

    def __init__(
            self, 
            quality_classifier, 
            transition_model, 
            #history_length, mode = 1,
        ):

        super(StateProcess1, self).__init__()

        self.clf = quality_classifier
        self.transition = transition_model
        #self.history_length = history_length
        #self.mode = mode

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        quality_hist = torch.stack(
            [self.clf(x[i].to('cpu')).unsqueeze(1) for i in range(x.size(0))], 
            dim = 0).to(x.device)
        x = torch.cat([x, quality_hist], dim = -1)
        
        if return_sample:
            delta = self.transition.predict(x)
            return x[:,-1,:-1] + delta
        
        else:
            delta_mu, sigma = self.transition.predict(x, return_sample = False)
            return x[:,-1,:-1] + delta_mu, sigma


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
    
    hp = {}
    hp['rat name'] = 'Bon'
    hp['input HL'], hp['label HL'] = 0, 0
    hp['include vel'], hp['dirvel'] = False, False
    hp['dmin'], hp['dmax'] = 0.5, 20
    hp['resolution'], hp['threshold'] = 0.1, 100
    hp['presence'], hp['p_range'] = False, 2

    
    data = generate_dataset(
        rat_name = hp['rat name'], 
        input_history_length = hp['input HL'], spike_bin_size = 1, 
        label_history_length = hp['label HL'], include_velocity = hp['include vel'], 
        dirvel = hp['dirvel'], dmin = hp['dmin'], dmax = hp['dmax'],
        grid_resolution = hp['resolution'], balance_resolution = hp['resolution'], 
        threshold = hp['threshold'], presence = hp['presence'], p_range = hp['p_range'],
        )

    in_train_input, in_train_label = data['train_positions'].float(), data['train_labels'].float()
    valid_input, valid_label = data['valid_positions'].float(), data['valid_labels'].float()
    test_input, test_label = data['test_positions'].float(), data['test_labels'].float()
    
    maze_grid = data['maze_grid']
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    

    tf = RangeNormalize(dim = 2, norm_mode = [0,0,])
    tf.fit(
        range_min = (xmin, ymin,), range_max = (xmax, ymax,),
        )
    
    insideMaze = GridClassifier(
        grid = maze_grid, 
        xmin = xmin, ymin = ymin, resolution = hp['resolution'], 
        transform = tf,
    )


    normal = Normal(torch.zeros((1,)).squeeze(), torch.ones((1,)).squeeze() * 0.1)

    iti = [in_train_input]
    itl = [in_train_label]
    vi = [valid_input]
    vl = [valid_label]

    print('\nupsampling in-maze data.....')
    for _ in range(150000 // in_train_input.size(0)):
        iti.append(in_train_input + normal.sample(in_train_input.size()))
        itl.append(in_train_label + normal.sample(in_train_label.size()))
        vi.append(valid_input + normal.sample(valid_input.size()))
        vl.append(valid_label + normal.sample(valid_label.size()))
    
    iti = torch.cat(iti, dim = 0)
    vi = torch.cat(vi, dim = 0)


    in_train_input = torch.cat([
        iti, torch.ones((iti.size(0), iti.size(1), 1)),
    ], dim = -1)
    in_train_label = torch.cat(itl, dim = 0)

    valid_input = torch.cat([
        vi, torch.ones((vi.size(0), vi.size(1), 1)),
    ], dim = -1)
    valid_label = torch.cat(vl, dim = 0)

    test_input = torch.cat([
        test_input, torch.ones((test_input.size(0), test_input.size(1), 1)),
    ], dim = -1)


    print('\ngathering random walk data.....')
    out_data = generate_trajectory_history(
        data = torch.load(f'Datasets/random walk/{hp["rat name"]}.pt'), 
        history_length = hp['input HL'],
    )
    out_data = out_data[torch.randperm(out_data.size(0))[:in_train_input.size(0)]]

    out_train_input, out_train_label = out_data[:-1,:,:], out_data[1:,-1,:]

    out_train_input = generate_inside_maze_prob_history(
        data = out_train_input, inside_maze_model = insideMaze,
    )


    print('\nnormalizing inputs & labels.....')
    #train_input = torch.cat([in_train_input, out_train_input], dim = 0)
    train_input = in_train_input
    train_input[:,:,:-1] = tf.transform(train_input[:,:,:-1])
    #train_label = tf.transform(torch.cat([in_train_label, out_train_label], dim = 0)) - train_input[:,-1,:-1]
    train_label = tf.transform(in_train_label) - train_input[:,-1,:-1]

    valid_input[:,:,:-1] = tf.transform(valid_input[:,:,:-1])
    valid_label = tf.transform(valid_label) - valid_input[:,-1,:-1]

    test_input = tf.transform(test_input[:,:,:-1])

    print('\n', train_input.size(), valid_input.size(), test_input.size())
    

    hp['hidden layer sizes'] = [16,16]

    """ model_name = 'GaussianBayesMLP-param_dist-gaussian'
    JointProbXY = GaussianBayesMLP(
        hidden_layer_sizes = hp['hidden layer sizes'], 
        input_dim = train_input.size(1)*train_input.size(2),
        latent_dim = 4 if hp['include vel'] else 2, 
        parameter_distribution = 'gaussian',
    ) """

    model_name = 'GaussianMLP'
    transition_model = GaussianMLP(
        hidden_layer_sizes = hp['hidden layer sizes'],
        input_dim = train_input.size(1)*train_input.size(2),
        latent_dim = 4 if hp['include vel'] else 2,
    )

    """ model_name = 'GaussianLTC'
    JointProbXY = GaussianLTC(
        hidden_layer_size = 4, num_layers = 1,
        sequence_dim = train_input.size(1), feature_dim = train_input.size(2), 
        latent_dim = 4 if hp['include vel'] else 2,
        last_hidden_state = False, use_cell_memory = False, solver_unfolds = 6,
    ) """
    
    
    root = None
    root = 'StateModels/trained/GaussianMLP_2023-11-3_23-15-58'

    hp['lr'] = 1e-3

    if root == None:

        trainer = TrainerMLE(
            optimizer = torch.optim.SGD(transition_model.parameters(), lr = hp['lr']),
        )
        _, _, _, fig = trainer.train(
            model = transition_model,
            train_data = Data(train_input, train_label),
            valid_data = Data(valid_input, valid_label),
            epochs = 1000, batch_size = 512, plot_losses = True,
        )

        now = datetime.datetime.now()
        root = f'StateModels/trained/{model_name}_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        os.mkdir(root)

        torch.save(transition_model.state_dict(), root+'/state_dict.pt')

        fig.savefig(root+'/loss_curves.jpeg')
        plt.close(fig)

        state_process = StateProcess1(
            transition_model = transition_model,
            quality_classifier = insideMaze,
            )
    
    else:
        transition_model.load_state_dict(torch.load(root+'/state_dict.pt'))
        state_process = StateProcess1(
            transition_model = transition_model,
            quality_classifier = insideMaze,
            )
    
    state_process.to('cpu')

    # evaluate performance (MSE) on test data
    pred = tf.untransform(state_process.predict(test_input, return_sample = True))
    
    pred_MSE = nn.MSELoss()(pred, test_label)
    print('\ntest mse: %.3f' % pred_MSE)
    
    # plot performance on test data
    pred_mean, pred_sdev = state_process.predict(test_input, return_sample = False)
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_sdev**2)
    
    fig = plot_model_predictions(
        pred_mean, pred_vars, test_label, 
        title = f'P(X,Y|input) | HL = {hp["input HL"]}\nTest Predictions (MSE: %.3f)' % pred_MSE,
    )

    fig.savefig(root+'/test_predictions.jpeg')
    plt.close(fig)



