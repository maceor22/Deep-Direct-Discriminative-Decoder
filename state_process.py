import os
from copy import deepcopy
from maze_utils import *
from misc_plotting import *
from in_maze_model import GridClassifier, generate_inside_maze_prob_history
from data_generation import generate_dataset, generate_trajectory_history
from models import *
from trainers import *
from D4 import DeepDirectDiscriminativeDecoder5
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.distributions import Normal
import datetime


class StateProcess(nn.Module):

    def __init__(
            self, penalty_model, proposal_model,
            transition_model, transition_delta = False,
        ):

        super(StateProcess, self).__init__()

        self.penalty = penalty_model
        self.proposal = proposal_model
        self.transition = transition_model
        self.delta = transition_delta

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        Q = self.proposal(x)

        if return_sample:
            return Q.sample()
        else:
            return Q

    def prob(self, x_prev, x_curr):
        if self.delta:
            return self.transition.prob(x_prev, x_curr - x_prev[:,-1,:])
        else:
            return self.transition.prob(x_prev, x_curr)



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



def train_state_process(
        D4, train_data, valid_data,
        learning_rate, grad_clip_value,
        num_train_batches = 10, num_valid_batches = 20,
        epochs = 100, batch_size = 256, 
        plot_losses = False, suppress_prints = False,
        ):

    train_batches = []
    lower = 0
    upper = batch_size

    while upper <= len(train_data):    
        train_batches.append(train_data[lower:upper])
        lower += batch_size
        upper += batch_size
    
    valid_batches = []
    lower = 0
    upper = batch_size

    while upper <= len(valid_data):
        valid_batches.append(valid_data[lower:upper])
        lower += batch_size
        upper += batch_size

    train_losses = []
    valid_losses = []

    optimizer = torch.optim.SGD(D4.parameters(), lr = learning_rate)

    best_loss = 1e10
    stime = time.time()

    for epoch in range(1, epochs+1):

        train_ix = torch.randperm(len(train_batches))[:num_train_batches]
        valid_ix = torch.randperm(len(valid_batches))[:num_valid_batches]

        tr_input = []
        tr_label = []
        for ix in train_ix:
            tr_input.append(train_batches[ix][0])
            tr_label.append(train_batches[ix][1])
        
        va_input = []
        va_label = []
        for ix in valid_ix:
            va_input.append(valid_batches[ix][0])
            va_label.append(valid_batches[ix][1])
        
        tr_input = torch.cat(tr_input, dim = 0)
        tr_label = torch.cat(tr_label, dim = 0)

        va_input = torch.cat(va_input, dim = 0)
        va_label = torch.cat(va_label, dim = 0)

        trainer = TrainerMLE(optimizer = optimizer, suppress_prints = True)
        _, train_loss, valid_loss = trainer.train(
            D4,
            train_data = Data(tr_input, tr_label),
            valid_data = Data(va_input, va_label),
            grad_clip_value = grad_clip_value,
            epochs = 1, batch_size = batch_size, shuffle = False,
        )

        train_losses = train_losses + train_loss
        valid_losses = valid_losses + valid_loss

        if valid_loss[0] < best_loss:
            best_loss = valid_loss[0]
            best_epoch = epoch
            best_state_dict = deepcopy(D4.state_dict())
                
        if not suppress_prints:
            # printing
            if epoch % 10 == 0:
                print(f'------------------{epoch}------------------')
                print('training loss: %.2f | validation loss: %.2f' % (
                    train_loss[0], valid_loss[0]))
                
                time_elapsed = (time.time() - stime)
                pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
                
                print('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                    time_elapsed, pred_time_remaining))
        
    # load best model
    D4.load_state_dict(best_state_dict)
    
    # plot loss curves
    if plot_losses:
        
        Train_losses = np.array(train_losses)
        Train_losses[np.where(Train_losses > 100)] = 100
        
        Valid_losses = np.array(valid_losses)
        Valid_losses[np.where(Valid_losses > 100)] = 100
        
        fig = plt.figure()
        
        plt.plot(range(1,epochs+1), Train_losses, '0.4', label = 'training')
        plt.plot(range(1,epochs+1), Valid_losses, 'b', label = 'validation')
        
        plt.xlabel('Epochs')
        plt.xticks(range(0,epochs+1,int(epochs // 10)))
        plt.ylabel('Negative Log Likelihood')
        plt.title('Loss Curves')
        plt.legend()

        return best_epoch, train_losses, valid_losses, fig
        
    return best_epoch, train_losses, valid_losses



if __name__ == '__main__':
    
    hp = {}
    hp['rat name'] = 'Bon'
    hp['pos HL'], hp['spike HL'], hp['label HL'] = 0, 8, 0
    hp['include vel'], hp['dirvel'] = False, False
    hp['dmin'], hp['dmax'] = 0.0, 100
    hp['grid resolution'], hp['balance resolution'] = 2, 0.1
    hp['threshold'], hp['presence'], hp['p_range'] = 100, False, 2
    hp['downsample'], hp['upsample'] = True, False

    
    data = generate_dataset(
        rat_name = hp['rat name'], pos_history_length = hp['pos HL'], 
        spike_history_length = hp['spike HL'], spike_bin_size = 1, 
        label_history_length = hp['label HL'], include_velocity = hp['include vel'], 
        dirvel = hp['dirvel'], dmin = hp['dmin'], dmax = hp['dmax'],
        grid_resolution = hp['grid resolution'], balance_resolution = hp['balance resolution'], 
        threshold = hp['threshold'], presence = hp['presence'], p_range = hp['p_range'],
        down_sample = hp['downsample'], up_sample = hp['upsample'],
        )
    
    bal_train_pos = data['bal_train_positions'].float()
    bal_train_label = data['bal_train_labels'].float()

    train_spikes = data['raw_train_spikes'].float()
    train_label = data['raw_train_labels'].float()

    valid_pos = data['raw_valid_positions'].float()
    valid_spikes = data['raw_valid_spikes'].float()
    valid_label = data['raw_valid_labels'].float()

    test_pos = data['raw_test_positions'].float()
    test_spikes = data['raw_test_spikes'].float()
    test_label = data['raw_test_labels'].float()


    maze_grid = data['maze_grid']
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    

    tf = RangeNormalize(dim = 2, norm_mode = [0,0,])
    tf.fit(
        range_min = (xmin, ymin,), range_max = (xmax, ymax,),
        )
    
    insideMaze = GridClassifier(
        grid = maze_grid, 
        xmin = xmin, ymin = ymin, resolution = hp['grid resolution'], 
        transform = tf,
    )

    print('\nnormalizing inputs & labels.....')

    bal_train_pos = tf.transform(bal_train_pos)
    bal_train_label = tf.transform(bal_train_label)

    train_spikes = torch.log(train_spikes + 1)
    train_label = tf.transform(train_label)

    valid_pos = tf.transform(valid_pos)
    valid_spikes = torch.log(valid_spikes + 1)
    valid_label = tf.transform(valid_label)
    
    test_pos = tf.transform(test_pos)
    test_spikes = torch.log(test_spikes + 1)


    print('\n', train_label.size(), valid_label.size(), test_label.size())
    

    """ hp['hidden layer sizes'] = [8,8]
    hp['state covariance type'] = 'diag'

    model_name = 'GaussianTransitionMLP'
    transition_model = GaussianTransitionMLP(
        hidden_layer_sizes = hp['hidden layer sizes'],
        input_dim = (hp['pos HL'] + 1) * train_label.size(1),
        latent_dim = 4 if hp['include vel'] else 2,
        #covariance_type = hp['state covariance type'],
        #dropout_p = 0,
    ) """
        

    obs_root = 'ObservationModels/trained/MVN-Mix-MLP_2024-1-31_11-59-34'
    prediction_model = GaussianMixtureMLP(
        hidden_layer_sizes = [24, 24], 
        num_mixtures = 5, 
        input_dim = train_spikes.size(1) * train_spikes.size(2), 
        latent_dim = 2,
        covariance_type = 'full',
        )
    prediction_model.load_state_dict(torch.load(obs_root+'/P_X__Y_H/state_dict.pt'))

    
    root = None

    hp['lr'] = 1e-3

    if root == None:

        """ print('\ntraining transition likelihood model.....')
        trainer = TrainerMLE(
            optimizer = torch.optim.SGD(transition_model.parameters(), lr = hp['lr']),
            print_every = 100,
        )
        _, _, _, pretrain_fig = trainer.train(
            model = transition_model,
            train_data = Data(bal_train_pos, bal_train_label),
            valid_data = Data(valid_pos, valid_label),
            grad_clip_value = 5,
            epochs = 1000, batch_size = 512, plot_losses = True,
        ) """

        D4 = DeepDirectDiscriminativeDecoder5(
            prediction_model = prediction_model,
            penalty_model = insideMaze,
            latent_dim = train_label.size(1), 
            state_covariance_type = 'diag',
            transition_variance = (train_label[1:] - train_label[:-1]).var(0),
            proposal_variance_init = 2 * (train_label[1:] - train_label[:-1]).var(0),
            )
        
        print('\ntraining proposal distribution parameters.....\n')
        _, _, _, fig = train_state_process(
            D4, 
            train_data = Data(train_spikes, train_label),
            valid_data = Data(valid_spikes, valid_label),
            learning_rate = hp['lr'], grad_clip_value = 5,
            num_train_batches = 30, num_valid_batches = 15,
            epochs = 100, batch_size = 512, plot_losses = True,
        )

        now = datetime.datetime.now()
        root = f'StateModels/trained/inq_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        os.mkdir(root)

        #torch.save(transition_model.state_dict(), root+'/transition_state_dict.pt')
        torch.save(D4.state_dict(), root+'/decoder_state_dict.pt')

        #pretrain_fig.savefig(root+'/pretrain_loss_curves.jpeg')
        fig.savefig(root+'/train_loss_curves.jpeg')
        #plt.close(pretrain_fig)
        plt.close(fig)

    
    else:
        transition_model.load_state_dict(torch.load(root+'/state_dict.pt'))
        """ state_process = StateProcess1(
            transition_model = transition_model,
            quality_classifier = insideMaze,
            ) """
    
    #state_process.to('cpu')
    transition_model.to('cpu')
    
    # plot performance on test data
    """ pred_mean, pred_sdev = state_process.predict(test_input, return_sample = False)
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_sdev**2) """

    pred_distribution = transition_model.predict(test_pos, return_sample = False)

    n_samples = 1000
    pred_samples = tf.untransform(pred_distribution.sample((n_samples,)))
    """ pred_samples = tf.untransform(
        test_input[:,-1,:].expand(n_samples,-1,-1) + pred_distribution.sample((n_samples,))) """
    error = pred_samples - test_label.expand(n_samples,-1,-1)
    pred_MSE = (error**2).mean(0).mean(0).mean(0)
    print('\ntest mse: %.3f' % pred_MSE)

    if type(pred_distribution) == torch.distributions.normal.Normal:
        pred_mean = pred_distribution.loc #+ test_input[:,-1,:]
        pred_vars = pred_distribution.scale**2
    elif type(pred_distribution) == torch.distributions.multivariate_normal.MultivariateNormal:
        pred_mean = pred_distribution.loc #+ test_input[:,-1,:]
        pred_covar = pred_distribution.scale_tril @ pred_distribution.scale_tril.transpose(1,2)
        pred_vars = torch.cat(
            [pred_covar[:,0,0].unsqueeze(1), pred_covar[:,1,1].unsqueeze(1)], dim = 1)
    
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_vars)    
    
    title = 'State Transition Likelihood Model '
    title += 'P(X\u2096|X\u2096\u208B\u2081)'
    title += '\nTest Predictions (RMSE: %.3f)' % pred_MSE**0.5
    fig = plot_model_predictions(
        pred_mean, pred_vars, test_label, title = title,
    )

    fig.savefig(root+'/test_predictions.jpeg')
    plt.close(fig)



