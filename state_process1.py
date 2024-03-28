import os
from copy import deepcopy
from maze_utils import *
from misc_plotting import *
from in_maze_model import GridClassifier, generate_inside_maze_prob_history
from data_generation import generate_dataset, generate_trajectory_history
from models import *
from trainers import *
from D4 import DeepDirectDiscriminativeDecoder2
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
import datetime



# random walk model
class GaussianTransition(nn.Module):
    
    def __init__(self, latent_dim, init_scale_tril = None, covariance_type = 'diag'):
        super(GaussianTransition, self).__init__()
        
        if covariance_type == 'diag':
            tril_components = torch.zeros((latent_dim,))
        elif covariance_type == 'full':
            tril_components = torch.zeros((((latent_dim+1) * latent_dim)/2,))

        if init_scale_tril == None:
            tril_components = tril_components.uniform_()
        else:
            if covariance_type == 'diag':
                for i in range(latent_dim):
                    tril_components[i] = init_scale_tril[i,i].log()
            
            elif covariance_type == 'full':
                idx = 0
                for i in range(latent_dim):
                    for j in range(i+1):
                        if i == j:
                            tril_components[idx] = init_scale_tril[i,j].log()
                        else:
                            tril_components[idx] = init_scale_tril[i,j]
                        idx += 1
        
        self.epsilon = 1e-20
        self.latent_dim = latent_dim
        self.tril_components = nn.Parameter(tril_components, requires_grad = True)
        self.covar_type = covariance_type
    
    def format_cholesky_tril(self, tril_components):
        batch_shape = tril_components.shape[:-1]
        cholesky_tril = torch.zeros(
            (*[_ for _ in batch_shape], self.latent_dim, self.latent_dim),
            device = tril_components.device,
            )
        
        if self.covar_type == 'diag':
            for i in range(self.latent_dim):
                cholesky_tril[...,i,i] = tril_components[...,i].exp() + self.epsilon

        elif self.covar_type == 'full':
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        cholesky_tril[...,i,j] = tril_components[...,idx].exp() + self.epsilon
                    else:
                        cholesky_tril[...,i,j] = tril_components[...,idx]
                    idx += 1
        
        return cholesky_tril
    

    def forward(self, x):
        scale_tril = self.format_cholesky_tril(
            self.tril_components.expand(x.size(0),-1)).to(x.device)
        return MultivariateNormal(loc = x[...,-1,:], scale_tril = scale_tril)
        
    def predict(self, x, return_sample = True):
        mvn = self.forward(x)
        
        if return_sample:
            return mvn.sample()
        else:
            return mvn
    
    def log_prob(self, x, y):
        mvn = self.forward(x)
        return mvn.log_prob(y)



class GaussianTransitionMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            covariance_type = 'diag', fixed_mean = True, 
            dropout_p = 0, epsilon = 1e-20,
            ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianTransitionMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Dropout(p = dropout_p))
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        if covariance_type == 'diag':
            num_tril_components = latent_dim
        elif covariance_type == 'full':
            num_tril_components = int(latent_dim*(latent_dim+1)/2)
        
        self.lin = nn.Sequential(*layers)

        self.tril_components = nn.Sequential(
            nn.Dropout(p = dropout_p), 
            nn.Linear(layer_sizes[-1], num_tril_components),
            )
        
        if not fixed_mean:
            self.mu_delta = nn.Sequential(
                nn.Dropout(p = dropout_p),
                nn.Linear(layer_sizes[-1], latent_dim),
            )
        
        self.fixed_mean = fixed_mean
        self.latent_dim = latent_dim
        self.covar_type = covariance_type
        self.epsilon = epsilon

    # ...
    def format_cholesky_tril(self, tril_components):
        batch_shape = tril_components.shape[:-1]
        cholesky_tril = torch.zeros(
            (*[_ for _ in batch_shape], self.latent_dim, self.latent_dim),
            device = tril_components.device,
            )
        
        if self.covar_type == 'diag':
            for i in range(self.latent_dim):
                cholesky_tril[...,i,i] = tril_components[...,i].exp() + self.epsilon

        elif self.covar_type == 'full':
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        cholesky_tril[...,i,j] = tril_components[...,idx].exp() + self.epsilon
                    else:
                        cholesky_tril[...,i,j] = tril_components[...,idx]
                    idx += 1
        
        return cholesky_tril
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        temp = self.lin(x.flatten(-2,-1))
        cholesky_tril = self.format_cholesky_tril(self.tril_components(temp))
        return MultivariateNormal(
            loc = x[...,-1,:] if self.fixed_mean else x[...,-1,:] + self.mu_delta(temp), 
            scale_tril = cholesky_tril,
            )
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        mvn = self.forward(x)
        
        if return_sample:
            return mvn.sample()
        else:
            return mvn
        
    
    def log_prob(self, x, y):
        mvn = self.forward(x)
        return mvn.log_prob(y)



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



def cross_validate(
        prediction_model, tf, batch_size,
        data_input, data_label, covariance_multipliers, 
        num_folds = 10, covariance_type = 'diag', 
        transition_model = None, proposal_model = None,
        penalty_model = None, plotting = False,
        ):
    print('\n\n')
    cov = (data_label[1:] - data_label[:-1]).T.cov()

    ret_mean = torch.zeros((len(covariance_multipliers),))
    ret_sdev = torch.zeros_like(ret_mean)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stime = time.time()

    for i in range(len(covariance_multipliers)):
        print('multiplier: ', covariance_multipliers[i])
        
        if transition_model == None:
            transition = GaussianTransition(
                latent_dim = data_label.size(1),
                init_scale_tril = torch.linalg.cholesky(cov * covariance_multipliers[i]),
                covariance_type = covariance_type,
            )
            DDD = DeepDirectDiscriminativeDecoder2(
                latent_dim = data_label.size(1), prediction_model = prediction_model,
                transition_model = transition, proposal_model = proposal_model,
                penalty_model = penalty_model,
            )
        
        else:
            proposal = GaussianTransition(
                latent_dim = data_label.size(1),
                init_scale_tril = torch.linalg.cholesky(cov * covariance_multipliers[i]),
                covariance_type = covariance_type,
            )
            DDD = DeepDirectDiscriminativeDecoder2(
                latent_dim = data_label.size(1), prediction_model = prediction_model,
                transition_model = transition_model, proposal_model = proposal, 
                penalty_model = penalty_model,
            )

        DDD = DDD.to(device)

        upper = data_input.size(0)
        lower = upper - batch_size
        rmse = torch.zeros((num_folds,))

        for k in range(num_folds):
            filter_trace = DDD.filter(
                data_input[lower:upper].to(device), n_particles = 200).detach().to('cpu')

            rmse[k] = nn.MSELoss()(
                tf.untransform(filter_trace), 
                tf.untransform(data_label[lower:upper].expand(DDD.n_particles,-1,-1))).item()**0.5

            lower -= batch_size
            upper -= batch_size
        
        ret_mean[i] = rmse.mean(0)
        ret_sdev[i] = rmse.std(0)
        print('mean: %.3f , sdev: %.3f | time elapsed: %.1f min' % (ret_mean[i], ret_sdev[i], (time.time()-stime)/60))

    if plotting:
        fig, ax = plt.subplots()
        
        plt.xlabel('Covariance Multiplier')
    
        ax.plot(covariance_multipliers, ret_mean, 'navy', label = 'pred mean')
        ax.fill_between(
            covariance_multipliers,
            ret_mean + 2*ret_sdev,
            ret_mean - 2*ret_sdev,
            color = 'cyan',
            label = 'pred 2*sdev',
            )
        ax.legend(fontsize = 6, loc = 2)
        ax.set_ylabel('Root Mean Squared Error')
        
        fig.suptitle('DDD Filter Cross-Validation\n95% Confidence Interval')

        return ret_mean, ret_sdev, fig
    
    else:
        return ret_mean, ret_sdev


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

    #print((valid_label[1:] - valid_label[:-1]).T.cov())
    valid_pos = tf.transform(valid_pos)
    valid_spikes = torch.log(valid_spikes + 1)
    valid_label = tf.transform(valid_label)
    #print((valid_label[1:] - valid_label[:-1]).T.cov())
    
    test_pos = tf.transform(test_pos)
    test_spikes = torch.log(test_spikes + 1)


    print('\n', train_label.size(), valid_label.size(), test_label.size())
        

    obs_root = 'ObservationModels/trained/MVN-Mix-MLP_2024-2-1_1-48-9'
    prediction_model = GaussianMixtureMLP(
        hidden_layer_sizes = [32, 32], 
        num_mixtures = 5, 
        input_dim = train_spikes.size(1) * train_spikes.size(2), 
        latent_dim = 2,
        covariance_type = 'full',
        )
    prediction_model.load_state_dict(torch.load(obs_root+'/P_X__Y_H/state_dict.pt'))


    hp['hidden layer sizes'] = [4,4]
    hp['state covariance type'] = 'diag'

    model_name = 'GaussianTransitionMLP'
    transition_model = GaussianTransitionMLP(
        hidden_layer_sizes = hp['hidden layer sizes'],
        input_dim = (hp['pos HL'] + 1) * train_label.size(1),
        latent_dim = 4 if hp['include vel'] else 2,
        covariance_type = hp['state covariance type'],
        fixed_mean = True,
        dropout_p = 0,
    )

    """ model_name = 'GaussianMLP_objectiveP(Xk;Hk)_'
    transition_model = GaussianMLP(
        hidden_layer_sizes = hp['hidden layer sizes'],
        input_dim = (hp['pos HL'] + 1) * train_label.size(1),
        latent_dim = 4 if hp['include vel'] else 2,
        covariance_type = hp['state covariance type'],
        dropout_p = 0,
    ) """

    
    root = None
    root = 'StateModels/trained/GaussianTransitionMLP2024-3-25_14-11-12'

    hp['lr'] = 1e-3

    if root == None:

        print('\ntraining state transition model.....\n')
        trainer = TrainerMLE(
            optimizer = torch.optim.SGD(transition_model.parameters(), lr = 1e-3)
        )
        _, _, _, train_fig = trainer.train(
            transition_model,
            train_data = Data(bal_train_pos, bal_train_label),
            valid_data = Data(valid_pos, valid_label),
            grad_clip_value = 5,
            epochs = 200, batch_size = 512, plot_losses = True,
        )
        
       
        now = datetime.datetime.now()
        root = f'StateModels/trained/' + model_name
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        os.mkdir(root)

        torch.save(transition_model.state_dict(), root+'/transition_state_dict.pt')

        train_fig.savefig(root+'/train_loss_curves.jpeg')
        plt.close(train_fig)

    
    else:
        transition_model.load_state_dict(torch.load(root+'/transition_state_dict.pt'))
    

    """ transition = GaussianTransition(
        latent_dim = train_label.size(1),
        init_scale_tril = torch.linalg.cholesky(100 * (train_label[1:] - train_label[:-1]).T.cov()),
        covariance_type = 'diag',
    )

    _, _, cv_fig = cross_validate(
        prediction_model = prediction_model, tf = tf, batch_size = 3000,
        data_input = torch.cat([train_spikes, valid_spikes], dim = 0), 
        data_label = torch.cat([train_label, valid_label], dim = 0),
        covariance_multipliers = [5 * i for i in range(1,21)], num_folds = 10, 
        transition_model = transition, proposal_model = None,
        penalty_model = insideMaze, plotting = True,
    )
    cv_fig.savefig('D4 Eval/cross_val_zoomed_proposalDNN1.jpeg') """

    transition = GaussianTransition(
        latent_dim = train_label.size(1),
        init_scale_tril = torch.linalg.cholesky(50 * (train_label[1:] - train_label[:-1]).T.cov()),
        covariance_type = 'diag',
    )
    proposal = GaussianTransition(
        latent_dim = train_label.size(1),
        init_scale_tril = torch.linalg.cholesky(20 * (train_label[1:] - train_label[:-1]).T.cov()),
        covariance_type = 'diag',
    )
    DDD = DeepDirectDiscriminativeDecoder2(
        latent_dim = train_label.size(1), prediction_model = prediction_model,
        transition_model = proposal, proposal_model = transition_model,
        penalty_model = insideMaze,
    )
    
    """ D4.to('cpu')
    #print(D4.Q_tril_components.exp())
    
    # plot performance on test data
    pred_distribution = D4.transition(test_pos)

    n_samples = 1000
    pred_samples = tf.untransform(pred_distribution.sample((n_samples,)))
    error = pred_samples - test_label.expand(n_samples,-1,-1)
    pred_RMSE = (error**2).mean(0).mean(0).mean(0)**0.5
    print('\ntransition model test RMSE: %.3f' % pred_RMSE)

    pred_mean = pred_distribution.loc
    pred_covar = pred_distribution.scale_tril @ pred_distribution.scale_tril.transpose(1,2)
    pred_vars = torch.cat(
        [pred_covar[:,0,0].unsqueeze(1), pred_covar[:,1,1].unsqueeze(1)], dim = 1)
    
    pred_mean, pred_vars = tf.untransform(pred_mean, pred_vars)    
    
    title = 'State Transition Likelihood Model '
    title += 'P(X\u2096|X\u2096\u208B\u2081)'
    title += '\nTest Predictions (RMSE: %.3f)' % pred_RMSE
    fig = plot_model_predictions(
        pred_mean, pred_vars, test_label, title = title,
    )

    fig.savefig(root+'/transition_test_predictions.jpeg')
    plt.close(fig) """


    print('\nbeginning test performance evaluation.....')
    rmse = []
    for _ in range(10):
        DDD = DDD.to('cuda')
        filter_trace = DDD.filter(
            observations = test_spikes.to('cuda'), n_particles = 5000)
        DDD = DDD.to('cpu')        
        filter_trace = tf.untransform(filter_trace.to('cpu'))

        rmse.append(nn.MSELoss()(
            filter_trace, test_label.expand(DDD.n_particles,-1,-1)).item()**0.5)
        print(rmse[-1])
    
    rmse = torch.tensor(rmse)
    rmse_mean = rmse.mean(dim = 0).item()
    rmse_ci = 2*rmse.std(dim = 0).item()

    filter_fig = plot_model_predictions(
        pred_mean = filter_trace.mean(dim = 0), 
        pred_vars = filter_trace.var(dim = 0),
        label_data = test_label, 
        title = 'Filter Performance | RMSE: %.3f +/- %.3f' % (rmse_mean, rmse_ci),
        )
    
    filter_fig.savefig('D4 Eval/filter_test_proposalDNN.jpeg')
    plt.close(filter_fig)

    print('\nmaking filter video.....')
    filter_vid = plot_filter_performance(
        DDD, test_label, tf, rmse = rmse[-1]
    )
    filter_vid.save('D4 Eval/filter_test_proposalDNN.mp4')

    """ print('video.....')

    LL_fig, LL_vid, _ = probabilistic_model_likelihood(
        model = prediction_model, model_name = 'P(X|Y,H)', 
        input_data = test_spikes[:], label_data = test_label[:], 
        posNorm = tf, untransform_label = False,
        grid_dim = 100, plotting = True, video = True,
        )
    LL_vid.save('D4 Eval/inq.mp4') """



