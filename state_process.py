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
import pandas as pd
           

""" class StateProcess(nn.Module):

    def __init__(
            self, quality_classifier, transition_model, 
            scale_multiplier = 1,
        ):

        super(StateProcess, self).__init__()

        self.clf = quality_classifier
        self.transition = transition_model
        self.scale_multiplier = scale_multiplier

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        Q = self.transition(x)
        if type(Q) == torch.distributions.normal.Normal:
            Q.scale = Q.scale * self.scale_multiplier
        elif type(Q) == torch.distributions.multivariate_normal.MultivariateNormal:
            Q.scale_tril = Q.scale_tril * self.scale_multiplier

        if return_sample:
            return Q.sample()
        else:
            return Q

    def prob(self, x, y):
        return self.transition.prob(x, y) """

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
            distribution = self.transition.predict(x, return_sample = False)
            distribution.loc += x[:,-1,:-1]
            return distribution


class StateProcess2(nn.Module):

    def __init__(
            self, 
            quality_classifier, 
            transition_model, 
            proposal,
            #history_length, mode = 1,
        ):

        super(StateProcess2, self).__init__()

        self.clf = quality_classifier
        self.transition = transition_model
        self.proposal = proposal
        #self.history_length = history_length
        #self.mode = mode

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        x = self.proposal.predict(x).unsqueeze(1)

        quality_hist = torch.stack(
            [self.clf(x[i].to('cpu')).unsqueeze(1) for i in range(x.size(0))], 
            dim = 0).to(x.device)
        x = torch.cat([x, quality_hist], dim = -1)
        
        if return_sample:
            delta = self.transition.predict(x)
            return x[:,-1,:-1] + delta
        
        else:
            distribution = self.transition.predict(x, return_sample = False)
            distribution.loc += x[:,-1,:-1]
            return distribution
        

class StateProcess3(nn.Module):

    def __init__(
            self, 
            quality_classifier, 
            transition_model, 
            proposal,
            epsilon = 1e-40,
        ):

        super(StateProcess3, self).__init__()

        self.clf = quality_classifier
        self.transition = transition_model
        self.proposal = proposal
        self.epsilon = epsilon

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        Q = self.proposal(x)
        
        if return_sample:
            return Q.sample()
        
        else:
            return Q
        
    def prob(self, x, y):
        quality_hist = torch.stack(
            [self.clf(x[i].to('cpu')).unsqueeze(1) for i in range(x.size(0))], 
            dim = 0).to(x.device)
        X = torch.cat([x, quality_hist], dim = -1)
        probs = self.transition.prob(X, y)
        print('\ntransition prob: ', probs)
        temp = self.proposal.prob(x, y)
        print('\nQ prob: ', temp)
        probs = probs / (temp + self.epsilon)
        return probs
    

class StateProcess4(nn.Module):

    def __init__(
            self, 
            quality_classifier, 
            transition_model, 
            proposal,
            epsilon = 1e-9,
        ):

        super(StateProcess4, self).__init__()

        self.clf = quality_classifier
        self.transition = transition_model
        self.proposal = proposal
        self.epsilon = epsilon

    def predict(self, x, return_sample = True):
        x = x.unsqueeze(0) if x.dim() == 2 else x

        quality_hist = torch.stack(
            [self.clf(x[i].to('cpu')).unsqueeze(1) for i in range(x.size(0))], 
            dim = 0).to(x.device)
        x = torch.cat([x, quality_hist], dim = -1)

        Q = self.transition(x)
        
        if return_sample:
            return x[:,-1,:-1] + Q.sample()
        
        else:
            return Q
        
    def prob(self, x, y):
        quality_hist = torch.stack(
            [self.clf(x[i].to('cpu')).unsqueeze(1) for i in range(x.size(0))], 
            dim = 0).to(x.device)
        X = torch.cat([x, quality_hist], dim = -1)
        probs = self.transition.prob(X, y - x[:,-1,:])
        #print('\ntransition prob: ', probs)
        temp = self.proposal.prob(x, y)
        #print('\nQ prob: ', temp)
        probs = probs / (temp + self.epsilon)
        return probs



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
    hp['rat name'] = 'Emile'
    hp['pos HL'], hp['spike HL'], hp['label HL'] = 0, 2, 0
    hp['include vel'], hp['dirvel'] = False, False
    hp['dmin'], hp['dmax'] = 0.05, 100
    hp['grid resolution'], hp['balance resolution'] = 2, 0.1
    hp['threshold'], hp['presence'], hp['p_range'] = 50, False, 2
    hp['downsample'] = True

    
    data = generate_dataset(
        rat_name = hp['rat name'], pos_history_length = hp['pos HL'], 
        spike_history_length = hp['spike HL'], spike_bin_size = 1, 
        label_history_length = hp['label HL'], include_velocity = hp['include vel'], 
        dirvel = hp['dirvel'], dmin = hp['dmin'], dmax = hp['dmax'],
        grid_resolution = hp['grid resolution'], balance_resolution = hp['balance resolution'], 
        threshold = hp['threshold'], presence = hp['presence'], p_range = hp['p_range'],
        down_sample = hp['downsample'],
        )

    train_input, train_label = data['bal_train_positions'].float(), data['bal_train_labels'].float()
    valid_input, valid_label = data['raw_valid_positions'].float(), data['raw_valid_labels'].float()
    test_input, test_label = data['raw_test_positions'].float(), data['raw_test_labels'].float()
    
    """ print((train_label - train_input[:,-1,:]).var(dim = 0))
    input('\npause') """

    """ temp = pd.DataFrame(
        data = torch.cat([in_train_input.squeeze(), in_train_label], dim = -1), 
        columns = ['Xk-1', 'Yk-1', 'Xk', 'Yk'])
    temp.to_csv('train.csv')

    temp = pd.DataFrame(
        data = torch.cat([valid_input.squeeze(), valid_label], dim = -1), 
        columns = ['Xk-1', 'Yk-1', 'Xk', 'Yk'])
    temp.to_csv('valid.csv')

    temp = pd.DataFrame(
        data = torch.cat([test_input.squeeze(), test_label], dim = -1), 
        columns = ['Xk-1', 'Yk-1', 'Xk', 'Yk'])
    temp.to_csv('test.csv') """


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


    normal = Normal(torch.zeros((1,)).squeeze(), torch.ones((1,)).squeeze() * 0.1)

    ti = [train_input]
    tl = [train_label]
    vi = [valid_input]
    vl = [valid_label]

    """ print('\nupsampling in-maze data.....')
    for _ in range(150000 // in_train_input.size(0)):
        iti.append(in_train_input + normal.sample(in_train_input.size()))
        itl.append(in_train_label + normal.sample(in_train_label.size()))
        vi.append(valid_input + normal.sample(valid_input.size()))
        vl.append(valid_label + normal.sample(valid_label.size())) """
    

    train_input = torch.cat(ti, dim = 0)
    train_label = torch.cat(tl, dim = 0)

    valid_input = torch.cat(vi, dim = 0)
    valid_label = torch.cat(vl, dim = 0)


    print('\nnormalizing inputs & labels.....')

    train_input = tf.transform(train_input)
    train_label = tf.transform(train_label) #- train_input[:,-1,:]

    valid_input = tf.transform(valid_input)
    valid_label = tf.transform(valid_label) #- valid_input[:,-1,:]

    #test_input = tf.transform(test_input[:,:,:-1])
    test_input = tf.transform(test_input)

    print('\n', train_input.size(), valid_input.size(), test_input.size())
    

    hp['hidden layer sizes'] = [16,16]
    hp['covariance type'] = 'diag'

    model_name = 'Gaussian-MLP'
    transition = GaussianMLP(
        hidden_layer_sizes = hp['hidden layer sizes'],
        input_dim = train_input.size(1)*train_input.size(2),
        latent_dim = 4 if hp['include vel'] else 2,
        covariance_type = hp['covariance type'],
    )
    
    
    root = None
    #root = 'StateModels/trained/GaussianMLP_2023-11-19_18-32-13'
    #root = 'StateModels/trained/GaussianTransitionMLP_2023-12-13_16-58-9'

    hp['lr'] = 1e-3

    if root == None:

        """ model = WishartWrapper(
            model = transition,
            df = torch.tensor(train_label.size(0)),
            prior = (train_label - train_input[:,-1,:]).var() * torch.eye(train_label.size(1)),
            model_weight = 0, wishart_weight = 1,
            )  
        
        pretrainer = TrainerMLE(
            optimizer = torch.optim.SGD(model.parameters(), lr = hp['lr']),
        )
        print('\npretraining transition model.....')
        _, _, _, pretrain_fig = pretrainer.train(
            model = model,
            train_data = Data(train_input, train_label),
            valid_data = Data(valid_input, valid_label),
            grad_clip_value = 5,
            epochs = 100, batch_size = 512, plot_losses = True,
        )

        model = WishartWrapper(
            model = transition,
            df = torch.tensor(train_label.size(0)),
            prior = (train_label - train_input[:,-1,:]).var() * torch.eye(train_label.size(1)),
            model_weight = 9, wishart_weight = 1,
            ) """

        trainer = TrainerMLE(
            optimizer = torch.optim.SGD(transition.parameters(), lr = hp['lr']),
        )
        print('\ntraining transition model.....')
        _, _, _, train_fig = trainer.train(
            model = transition,
            train_data = Data(train_input, train_label),
            valid_data = Data(valid_input, valid_label),
            grad_clip_value = 5,
            epochs = 200, batch_size = 512, plot_losses = True,
        )

        now = datetime.datetime.now()
        root = f'StateModels/trained/{model_name}_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        os.mkdir(root)

        torch.save(transition.state_dict(), root+'/state_dict.pt')

        #pretrain_fig.savefig(root+'/_pretrain_loss_curves.jpeg')
        train_fig.savefig(root+'/_train_loss_curves.jpeg')
        #plt.close(pretrain_fig)
        plt.close(train_fig)
    
    else:
        transition.load_state_dict(torch.load(root+'/state_dict.pt'))
        """ state_process = StateProcess1(
            transition_model = transition_model,
            quality_classifier = insideMaze,
            ) """
    
    #state_process.to('cpu')
    transition.to('cpu')
    
    # plot performance on test data
    """ pred_mean, pred_sdev = state_process.predict(test_input, return_sample = False)
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_sdev**2) """

    pred_distribution = transition.predict(test_input, return_sample = False)

    """ pred_samples = tf.untransform(pred_distribution.sample())
    pred_MSE = ((pred_samples - test_label)**2).mean(dim = 0).mean(dim = 0) """
    n_samples = 1000
    pred_samples = tf.untransform(pred_distribution.sample((n_samples,)))
    error = pred_samples - test_label.expand(n_samples,-1,-1)
    pred_MSE = (error**2).mean(0).mean(0).mean(0)
    print('\ntest mse: %.3f' % pred_MSE)

    if type(pred_distribution) == torch.distributions.normal.Normal:
        pred_mean = pred_distribution.loc
        pred_vars = pred_distribution.scale**2
    elif type(pred_distribution) == torch.distributions.multivariate_normal.MultivariateNormal:
        pred_mean = pred_distribution.loc
        pred_covar = pred_distribution.scale_tril @ pred_distribution.scale_tril.transpose(1,2)
        pred_vars = torch.cat(
            [pred_covar[:,0,0].unsqueeze(1), pred_covar[:,1,1].unsqueeze(1)], dim = 1)
    
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_vars)
    print(pred_vars[:200])

    """ for i in range(1500,1600):
        print('mean:', pred_mean[i],' variance:', pred_vars[i]) """
    
    title = 'State Transition Likelihood Model '
    title += 'P(X\u2096|X\u2096\u208B\u2081)'
    title += '\nTest Predictions (RMSE: %.3f)' % pred_MSE**0.5
    fig = plot_model_predictions(
        pred_mean, pred_vars, test_label, title = title,
    )

    fig.savefig(root+'/test_predictions.jpeg')
    plt.close(fig)



