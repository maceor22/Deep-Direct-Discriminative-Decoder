import os
from copy import deepcopy
import torch
import numpy as np
from torch import nn
from torch.distributions import Multinomial
from torch import autograd as ag
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from misc_plotting import *
from maze_utils import Maze, Data, RangeNormalize
from models import *
from trainers import *
from data_generation import generate_dataset, balance_dataset
import time
import datetime
#from memory_profiler import profile

class CountProcess(nn.Module):

    def __init__(self, count_model, n_trials, log_target = True):

        super(CountProcess, self).__init__()

        self.count_model = count_model
        self.n_trials = n_trials
        self.log_target = log_target
    
    def forward(self, x):
        return self.count_model(x)
    
    def predict(self, x):
        pred = self.count_model.predict(x).exp().ceil()
        
        if self.log_target:
            return torch.log(pred)
        else:
            return (pred - 1).long()
    
    def prob(self, x, target, n_samples = 50, reduction = 'mean'):
        if self.log_target:
            target = (target.exp() - 1).long()

        samples = torch.stack(
            [self.count_model.predict(x) for _ in range(n_samples)], dim = 1
            ).exp().ceil() - 1
        binomial_probs = samples.mean(dim = 1) / self.n_trials
        binomial_probs = torch.where(binomial_probs > 1, 1, binomial_probs)

        log_probs = torch.distributions.Binomial(
            total_count = self.n_trials, probs = binomial_probs).log_prob(target)
        
        if reduction == 'joint':
            probs = log_probs.sum(dim = 1).exp()
        elif reduction == 'mean':
            probs = log_probs.exp().mean(dim = 1)
        
        return probs
    
""" class P_X__H(nn.Module):

    def __init__(self, P_Y__H, P_X__Y_H):

        super(P_X__H, self).__init__()

        self.P_Y__H = P_Y__H
        self.P_X__Y_H = P_X__Y_H
    
    #@profile
    def prob(self, h, x, n_samples = 10):
        probs = torch.zeros((x.size(0),), device = x.device)

        y_distribution = self.P_Y__H(h)

        for _ in range(n_samples):
            y = y_distribution.sample().unsqueeze(1)
            probs += self.P_X__Y_H.prob(torch.cat([h, y], dim = 1), x)
            del y

        return probs """


""" class P_X__H(nn.Module):

    def __init__(self, transition_model):

        super(P_X__H, self).__init__()

        self.transition = transition_model
    
    #@profile
    def prob(self, x_prev, x_curr):
        if x_prev.dim() == 2:
            x_prev = x_prev.unsqueeze(1)
        
        probs = torch.zeros((x_curr.size(0),), device = x_curr.device)

        for n in range(x_curr.size(0)):
            probs[n] += self.transition.prob(
                x_prev, x_curr[n].expand(x_curr.size(0), -1)).sum(dim = 0)

        return probs """


class P_X__H(nn.Module):

    def __init__(self, transition_model, transition_delta = False):

        super(P_X__H, self).__init__()

        self.transition = transition_model
        self.delta = transition_delta
    
    #@profile
    def prob(self, x_prev, x_curr):
        if x_prev.dim() == 2:
            x_prev = x_prev.unsqueeze(1)
        
        x_prev = x_prev.expand(x_curr.size(0), x_prev.size(0), -1, -1)
        x_curr = x_curr.expand(x_prev.size(0), x_curr.size(0), -1).transpose(0,1)
        
        probs = self.transition.prob(
            x_prev, x_curr - x_prev[:,:,-1,:] if self.delta else x_curr
        ).sum(dim = -1)

        return probs


""" class P_X__H(nn.Module):

    def __init__(self, transition_model):

        super(P_X__H, self).__init__()

        self.transition = transition_model
    
    #@profile
    def prob(self, x_prev, x_curr, n_samples = 100):
        if x_prev.dim() == 2:
            x_prev = x_prev.unsqueeze(1)
        
        samples = x_prev[torch.randperm(x_prev.size(0))[:n_samples]]

        probs = self.transition.prob(
            samples.expand(x_curr.size(0), n_samples, -1, -1), 
            x_curr.expand(n_samples, x_curr.size(0), -1).transpose(0,1)
        ).sum(dim = -1)

        return probs   """  


""" class P_X__H(nn.Module):

    def __init__(self, transition_model):

        super(P_X__H, self).__init__()

        self.transition = transition_model
    
    #@profile
    def prob(self, x_prev, x_curr):
        if x_prev.dim() == 2:
            x_prev = x_prev.unsqueeze(1)
        
        Q = self.transition(x_prev)
        log_probs = Q.log_prob(x_curr)

        if type(Q) == torch.distributions.normal.Normal:
            log_probs = log_probs.sum(dim = -1)

        probs = log_probs.exp()
        del log_probs            
        return probs """



class ObservationProcess(nn.Module):

    def __init__(self, P_X__Y_H, P_X__H, epsilon = 1e-20):

        super(ObservationProcess, self).__init__()

        self.P_X__Y_H = P_X__Y_H
        self.P_X__H = P_X__H
        self.epsilon = epsilon
    
    def predict(self, y_curr, return_sample = True):
        return self.P_X__Y_H.predict(y_curr, return_sample)
    
    #@profile
    def prob(self, yh_curr, x_prev, x_curr):
        probs = self.P_X__Y_H.prob(yh_curr, x_curr)
        probs /= (self.P_X__H.prob(x_prev, x_curr) + self.epsilon)
        return probs
    

""" class ObservationProcess(nn.Module):

    def __init__(self, P_X__Y_H, P_X__H, epsilon = 1e-40):

        super(ObservationProcess, self).__init__()

        self.P_X__Y_H = P_X__Y_H
        self.P_X__H = P_X__H
        self.epsilon = epsilon
    
    def predict(self, y, return_sample = True):
        return self.P_X__Y_H.predict(y, return_sample)
    
    #@profile
    def prob(self, y, x):
        probs = self.P_X__Y_H.prob(y, x)**2
        probs /= (self.P_X__H.prob(y[:,:-1,:], x) + self.epsilon)
        return probs """



def density_mixture_history_evaluation(
        num_mixtures, hidden_layers,
        test_size, num_folds, 
        min_history, max_history, bin_size, step,
        pretrain_epochs, training_epochs,
        root = None, 
        HPD_alpha = 0.1, include_velocity = False, dirvel = False, plot = True,
    ):
    if root is None:
        now = datetime.datetime.now()
        root = f'ObservationModels/history eval/GMMLP_{num_mixtures}mix_hid{hidden_layers}_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        os.mkdir(root)

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

    history_lengths = np.arange(min_history, max_history+1, step)

    #train_acc = torch.zeros((len(history_lengths),num_folds))
    #train_area = torch.zeros((len(history_lengths),num_folds))
    #train_mse = torch.zeros((len(history_lengths),num_folds,3))

    test_acc = torch.zeros((len(history_lengths),num_folds))
    test_area = torch.zeros((len(history_lengths),num_folds))
    test_mse = torch.zeros((len(history_lengths),num_folds,3))

    idx = 0

    posNorm = RangeNormalize(dim = 2, norm_mode = [0,0,])
    posNorm.fit(
        range_min = (150, 50,), range_max = (270, 170,),
    )

    stime = time.time()

    for hl in history_lengths:
        hl_path = root + f'/HL{hl}'
        if not os.path.exists(hl_path):
            os.mkdir(hl_path)
        
        print(f'------------ HL: {hl} ------------')

        inputs = []
        labels = []
        for maze in mazes:
            new_input, _, new_label = maze.generate_data(
                input_history_length = hl, spike_bin_size = bin_size, 
                label_history_length = 0, shuffle = False,
                )
            inputs.append(new_input)
            labels.append(new_label)
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)
        
        spikeNorm = RangeNormalize(dim = inputs.size(2), norm_mode = 'auto')
        spikeNorm.fit(
            range_min = [0 for _ in range(inputs.size(2))], 
            range_max = [10 for _ in range(inputs.size(2))],
        )

        ix = torch.arange(inputs.size(0))
        
        upper = inputs.size(0)
        b1 = upper - test_size
        b0 = b1 - 2*test_size
        lower = test_size*(num_folds-1)
        
        for k in range(num_folds):
            train_ix = ix[lower:b0]
            valid_ix = ix[b0:b1]
            test_ix = ix[b1:upper]
            
            bal_ix = balance_dataset(
                label_data = labels[train_ix], 
                xmin = 150, xmax = 270, ymin = 50, ymax = 170, 
                resolution = 0.1, threshold = 100,
                )
            
            train_input = spikeNorm.transform(inputs[train_ix][bal_ix])
            valid_input = spikeNorm.transform(inputs[valid_ix])
            
            train_label = posNorm.transform(labels[train_ix][bal_ix])
            valid_label = posNorm.transform(labels[valid_ix])
            
            test_input = spikeNorm.transform(inputs[test_ix])
            test_label = labels[test_ix]

            JointProbXY = GaussianMixtureMLP(
                hidden_layer_sizes = hidden_layers, 
                num_mixtures = num_mixtures,
                input_dim = inputs.size(1)*inputs.size(2),
                latent_dim = 2,
            )

            fold_path = hl_path + f'/fold{k+1}'
            if not os.path.exists(fold_path):
                
                _, _, _, pretrain_fig = pretrain_density_mixture_network(
                    mixture_network = JointProbXY, covariance_type = 'diag',
                    train_input = train_input, train_label = train_label,
                    valid_input = valid_input, valid_label = valid_label,
                    epochs = pretrain_epochs, batch_size = 256, 
                    plot_losses = True, suppress_prints = True,
                )

                trainer = TrainerMLE(
                    optimizer = torch.optim.SGD(JointProbXY.parameters(), lr = 1e-3),
                    suppress_prints = True,
                )

                _, _, _, train_fig = trainer.train(
                    model = JointProbXY, 
                    train_data = Data(train_input, train_label), 
                    valid_data = Data(valid_input, valid_label),
                    epochs = training_epochs, batch_size = 256, plot_losses = True,
                )

                os.mkdir(fold_path)
                torch.save(JointProbXY.state_dict(), fold_path+'/state_dict.pt')
                pretrain_fig.savefig(fold_path+'/losses_pretrain.jpeg')
                train_fig.savefig(fold_path+'/losses_train.jpeg')
                plt.close(pretrain_fig)
                plt.close(train_fig)

            JointProbXY.load_state_dict(torch.load(fold_path+'/state_dict.pt'))
            
            """ tr_accuracy, tr_area_ratio = density_mixture_HPD(
                JointProbXY, covariance_type = 'diag',
                input_data = train_input, label_data = train_label, posNorm = posNorm,
                untransform_label = True, alpha = HPD_alpha, plotting = False,
            ) """
            fig, te_accuracy, te_area_ratio = density_mixture_HPD(
                JointProbXY, covariance_type = 'diag',
                input_data = test_input, label_data = test_label, posNorm = posNorm,
                untransform_label = False, alpha = HPD_alpha, plotting = True,
            )
            fig.savefig(fold_path+'/test_HPD.png')
            plt.close(fig)

            
            """ tr_pi, tr_mu, _ = JointProbXY(train_input)

            tr_pred0 = torch.zeros((train_input.size(0), train_label.size(1)))
            for i in range(train_input.size(0)):
                for k_ in range(num_mixtures):
                    tr_pred0[i,:] += tr_pi[i,k_] * tr_mu[i,k_,:]

            tr_pred1 = torch.zeros_like(tr_pred0)
            k_ = tr_pi.argmax(dim = 1)
            for i in range(k_.size(0)):
                tr_pred1[i,:] = tr_mu[i,k_[i],:]

            tr_pred2 = torch.zeros_like(tr_pred0)
            k_ = tr_pi.multinomial(num_samples = 1).squeeze(1)
            for i in range(k_.size(0)):
                tr_pred2[i,:] = tr_mu[i,k_[i],:] """


            te_pi, te_mu, _ = JointProbXY(test_input)

            te_pred0 = torch.zeros((test_input.size(0), test_label.size(1)))
            for i in range(test_input.size(0)):
                for k_ in range(num_mixtures):
                    te_pred0[i,:] += te_pi[i,k_] * te_mu[i,k_,:]

            te_pred1 = torch.zeros_like(te_pred0)
            k_ = te_pi.argmax(dim = 1)
            for i in range(k_.size(0)):
                te_pred1[i,:] = te_mu[i,k_[i],:]

            te_pred2 = torch.zeros_like(te_pred0)
            k_ = te_pi.multinomial(num_samples = 1).squeeze(1)
            for i in range(k_.size(0)):
                te_pred2[i,:] = te_mu[i,k_[i],:]


            """ train_acc[idx,k] = tr_accuracy
            train_area[idx,k] = tr_area_ratio """
            """ train_mse[idx,k,0] = nn.MSELoss()(
                posNorm.untransform(tr_pred0), posNorm.untransform(train_label))
            train_mse[idx,k,1] = nn.MSELoss()(
                posNorm.untransform(tr_pred1), posNorm.untransform(train_label))
            train_mse[idx,k,2] = nn.MSELoss()(
                posNorm.untransform(tr_pred2), posNorm.untransform(train_label)) """

            test_acc[idx,k] = te_accuracy
            test_area[idx,k] = te_area_ratio
            test_mse[idx,k,0] = nn.MSELoss()(posNorm.untransform(te_pred0), test_label)
            test_mse[idx,k,1] = nn.MSELoss()(posNorm.untransform(te_pred1), test_label)
            test_mse[idx,k,2] = nn.MSELoss()(posNorm.untransform(te_pred2), test_label)
            
            print(f'fold {k+1} completed | time elapsed (hrs): %.3f' % (
                (time.time() - stime)/3600))
            
            lower -= test_size
            b0 -= test_size
            b1 -= test_size
            upper -= test_size
        
        idx += 1
    
    #train_score = train_acc / train_area
    test_score = test_acc / test_area

    metrics_path = root + '/metrics'
    os.mkdir(metrics_path)

    torch.save(torch.from_numpy(history_lengths), metrics_path+'/history_lengths.pt')

    #torch.save(train_acc, metrics_path+'/train_accuracy.pt')
    #torch.save(train_area, metrics_path+'/train_area.pt')
    #torch.save(train_score, metrics_path+'/train_ratio.pt')
    #torch.save(train_mse, metrics_path+'/train_mse.pt')

    torch.save(test_acc, metrics_path+'/test_accuracy.pt')
    torch.save(test_area, metrics_path+'/test_area.pt')
    torch.save(test_score, metrics_path+'/test_ratio.pt')
    torch.save(test_mse, metrics_path+'/test_mse.pt')

    if plot:
        
        plot_density_mixture_metrics(root)
    
    return


def pretrain_density_mixture_network(
        mixture_network, fit_gmm, 
        train_input, train_label, valid_input, valid_label,
        grad_clip_value = 1, epochs = 100, batch_size = 256, 
        plot_losses = False, suppress_prints = False,
        ):
    gmm = fit_gmm
    
    train_z = torch.from_numpy(gmm.predict(train_label))
    valid_z = torch.from_numpy(gmm.predict(valid_label))
    
    mu = torch.from_numpy(gmm.means_).float()
    cov = torch.from_numpy(gmm.covariances_).float()

    if cov.dim() == 2:
        cov = cov * torch.eye(cov.size(1)).expand(cov.size(0),-1,-1)
    
    """ tr_pi_target = torch.zeros((train_z.size(0), mu.size(0)))
    tr_pi_target[torch.arange(train_z.size(0)),train_z] = 1 """
    
    tr_mu_target = torch.stack(
        [mu[train_z[i]] for i in range(train_z.size(0))], dim = 0
    )
    tr_cov_target = torch.stack(
        [cov[train_z[i]] for i in range(train_z.size(0))], dim = 0
    )
        
    """ va_pi_target = torch.zeros((valid_z.size(0), mu.size(0)))
    va_pi_target[torch.arange(valid_z.size(0)),valid_z] = 1 """
    
    va_mu_target = torch.stack(
        [mu[valid_z[i]] for i in range(valid_z.size(0))], dim = 0
    )
    va_cov_target = torch.stack(
        [cov[valid_z[i]] for i in range(valid_z.size(0))], dim = 0
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixture_network = mixture_network.to(device)
    
    train_losses = []
    valid_losses = []
    
    stime = time.time()
    
    best_loss = 1e10
    best_state_dict = deepcopy(mixture_network.state_dict())
    
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
            try:
                input_batch = ag.Variable(train_input[ix[lower:upper]].to(device))
                
                pi_target = ag.Variable(train_z[ix[lower:upper]].to(device))
                mu_target = ag.Variable(tr_mu_target[ix[lower:upper]].to(device))
                cov_target = ag.Variable(tr_cov_target[ix[lower:upper]].to(device))
                
                sample = MultivariateNormal(
                    loc = mu_target, covariance_matrix = cov_target).sample()

                distribution = mixture_network(input_batch)

                loss = Categorical(
                    probs = distribution.mixture_distribution.probs).log_prob(pi_target).exp()
                loss = (loss + mixture_network.log_prob(input_batch, sample).exp()) / 2
                loss = -torch.log(loss).mean(dim = 0)
                
                # zero gradients
                optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(mixture_network.parameters(), clip_value = grad_clip_value)
                # update weights
                optimizer.step()
                # aggregate training loss
                train_loss += loss.item()
            
            except:
                mixture_network.load_state_dict(best_state_dict)
                train_loss += 10
            
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
                try:
                    input_batch = valid_input[ix[lower:upper]].to(device)
                    
                    #pi_target = valid_z[ix[lower:upper]]
                    pi_target = valid_z[ix[lower:upper]].to(device)
                    mu_target = va_mu_target[ix[lower:upper]].to(device)
                    cov_target = va_cov_target[ix[lower:upper]].to(device)
                    
                    sample = MultivariateNormal(
                        loc = mu_target, covariance_matrix = cov_target).sample()

                    distribution = mixture_network(input_batch)

                    loss = Categorical(
                        probs = distribution.mixture_distribution.probs).log_prob(pi_target).exp()
                    loss = (loss + mixture_network.log_prob(input_batch, sample).exp()) / 2
                    loss = -torch.log(loss).mean(dim = 0)

                    # compute and aggregate validation loss 
                    valid_loss += loss.item()

                except:
                    mixture_network.load_state_dict(best_state_dict)
                    valid_loss += 10
                
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
        if epoch % 10 == 0 and not suppress_prints:
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
        
        fig = plt.figure()
        
        plt.plot(range(1,epochs+1), Train_losses, '0.4', label = 'training')
        plt.plot(range(1,epochs+1), Valid_losses, 'b', label = 'validation')
        
        plt.xlabel('Epochs')
        plt.xticks(range(0,epochs+1,int(epochs // 10)))
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Loss Curves')
        plt.legend()
        
        return best_epoch, train_losses, valid_losses, fig
    else:
        return best_epoch, train_losses, valid_losses    



if __name__ == '__main__':

    shl = 8
    phl = 0
    bs = 1
    
    data = generate_dataset(
        rat_name = 'Bon', pos_history_length = phl, 
        spike_history_length = shl, spike_bin_size = bs, label_history_length = 0, 
        include_velocity = False, dirvel = False, dmin = 0.0, dmax = 100,
        grid_resolution = 2, balance_resolution = 0.1, threshold = 100, 
        presence = False, p_range = 2, down_sample = False, up_sample = False,
        )

    bal_train_input, bal_train_label = data['raw_train_spikes'], data['raw_train_labels']
    raw_valid_input, raw_valid_label = data['raw_valid_spikes'], data['raw_valid_labels']
    
    raw_test_input, test_label = data['raw_test_spikes'], data['raw_test_labels'].float()
    
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    
    """ rem = [
        6, 11, 14, 24, 26, 27, 28, 29, 31, 32, 
        33, 34, 35, 37, 38, 41, 43, 46, 47, 48, 
        52, 53, 54, 56, 58, 59, 60, 61, 66, 68, 
        72, 74, 75, 76, 78, 79, 81, 83, 85, 86, 
        89, 90, 92, 
    ] """
    """ rem = [
        11, 14, 24, 26, 27, 28, 29, 31, 32, 33, 
        34, 35, 37, 38, 41, 43, 46, 47, 48, 52, 
        53, 54, 56, 58, 59, 60, 61, 66, 68, 72, 
        75, 76, 78, 81, 83, 85, 89, 
    ]

    bal_train_input = torch.stack(
        [bal_train_input[...,c] for c in range(bal_train_input.size(2)) 
         if c not in rem], dim = 2)
    raw_valid_input = torch.stack(
        [raw_valid_input[...,c] for c in range(raw_valid_input.size(2)) 
         if c not in rem], dim = 2)
    raw_test_input = torch.stack(
        [raw_test_input[...,c] for c in range(raw_test_input.size(2))
         if c not in rem], dim = 2) """


    """ plt.figure()
    plt.plot(bal_train_label[:,0], bal_train_label[:,1], 'o', color = '0.4', ms = 0.1)
    plt.show() """


    """ region_dict = {}

    region_dict[0] = {}
    region_dict[0]['xmin'], region_dict[0]['xmax'] = 15, 45
    region_dict[0]['ymin'], region_dict[0]['ymax'] = 145, 180

    region_dict[1] = {}
    region_dict[1]['xmin'], region_dict[1]['xmax'] = 45, 75
    region_dict[1]['ymin'], region_dict[1]['ymax'] = 145, 180

    region_dict[2] = {}
    region_dict[2]['xmin'], region_dict[2]['xmax'] = 75, 111
    region_dict[2]['ymin'], region_dict[2]['ymax'] = 145, 180

    region_dict[3] = {}
    region_dict[3]['xmin'], region_dict[3]['xmax'] = 111, 145
    region_dict[3]['ymin'], region_dict[3]['ymax'] = 145, 180

    region_dict[4] = {}
    region_dict[4]['xmin'], region_dict[4]['xmax'] = 145, 180
    region_dict[4]['ymin'], region_dict[4]['ymax'] = 145, 180

    region_dict[5] = {}
    region_dict[5]['xmin'], region_dict[5]['xmax'] = 180, 210
    region_dict[5]['ymin'], region_dict[5]['ymax'] = 145, 180

    region_dict[6] = {}
    region_dict[6]['xmin'], region_dict[6]['xmax'] = 15, 45
    region_dict[6]['ymin'], region_dict[6]['ymax'] = 105, 145

    region_dict[7] = {}
    region_dict[7]['xmin'], region_dict[7]['xmax'] = 45, 75
    region_dict[7]['ymin'], region_dict[7]['ymax'] = 105, 145

    region_dict[8] = {}
    region_dict[8]['xmin'], region_dict[8]['xmax'] = 75, 111
    region_dict[8]['ymin'], region_dict[8]['ymax'] = 105, 145

    region_dict[9] = {}
    region_dict[9]['xmin'], region_dict[9]['xmax'] = 111, 145
    region_dict[9]['ymin'], region_dict[9]['ymax'] = 105, 145

    region_dict[10] = {}
    region_dict[10]['xmin'], region_dict[10]['xmax'] = 145, 180
    region_dict[10]['ymin'], region_dict[10]['ymax'] = 105, 145

    region_dict[11] = {}
    region_dict[11]['xmin'], region_dict[11]['xmax'] = 180, 210
    region_dict[11]['ymin'], region_dict[11]['ymax'] = 105, 145

    region_dict[12] = {}
    region_dict[12]['xmin'], region_dict[12]['xmax'] = 0, 225
    region_dict[12]['ymin'], region_dict[12]['ymax'] = 75, 105

    region_dict[13] = {}
    region_dict[13]['xmin'], region_dict[13]['xmax'] = 190, 245
    region_dict[13]['ymin'], region_dict[13]['ymax'] = 0, 75


    plot_spikes_versus_regions(
        region_dict = region_dict, 
        spikes = data['bal_train_spikes'][:,-1,:],
        positions = data['bal_train_labels'],
        save_path = 'Exploratory/Emile'
        )
    input('pause.....')
 """

    train_input = torch.log(bal_train_input + 1)
    valid_input = torch.log(raw_valid_input + 1)
    test_input = torch.log(raw_test_input + 1)

    posNorm = RangeNormalize(dim = 2, norm_mode = [0,0,])
    posNorm.fit(
        range_min = (xmin, ymin,), range_max = (xmax, ymax,),
        )
    
    train_label = posNorm.transform(bal_train_label[:,:2]).float()
    valid_label = posNorm.transform(raw_valid_label[:,:2]).float()

    print('\n', train_input.size(), valid_input.size(), test_input.size())

    
    root = None
    root = 'ObservationModels/trained/MVN-Mix-MLP_2024-2-1_1-48-9'
    
    name = 'MVN-Mix-MLP'
    covar_type = 'full'
    discriminative = GaussianMixtureMLP(
        hidden_layer_sizes = [32,32],
        num_mixtures = 5,
        input_dim = train_input.size(1) * train_input.size(2),
        latent_dim = 2,
        covariance_type = covar_type,
        dropout_p = 0.2,
    )

    if root == None:
        
        gmm = GMM(
        n_components = discriminative.num_mixtures, 
        covariance_type = covar_type,
        tol = 1e-9, max_iter = 1000,
        )
        gmm.fit(train_label[torch.randperm(train_label.size(0))][:100000])


        print('\npre-training P(X|Y,H) .....\n')
        _, _, _, pretrained_fig = pretrain_density_mixture_network(
            mixture_network = discriminative, fit_gmm = gmm,
            train_input = train_input, train_label = train_label, 
            valid_input = valid_input, valid_label = valid_label,
            grad_clip_value = 5, epochs = 2000, batch_size = 512,
            plot_losses = True,
            )
        
        learned_fig = plot_pretrained_learned_mixtures(
            mixture_model = discriminative, 
            input_data = valid_input, label_data = valid_label,
            posNorm = posNorm,
        )

        now = datetime.datetime.now()
        root = f'ObservationModels/trained/{name}_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        path1 = root + '/P_X__Y_H'
        os.mkdir(root)
        os.mkdir(path1)

        trainer = TrainerMLE(
            optimizer = torch.optim.SGD(discriminative.parameters(), lr = 1e-3),
            print_every = 100,
            )
        
        print('\ntraining P(X|Y,H) .....\n')
        _, _, _, trained_fig = trainer.train(
            model = discriminative, 
            train_data = Data(train_input, train_label), 
            valid_data = Data(valid_input, valid_label),
            grad_clip_value = 5, epochs = 10000, batch_size = 512, 
            plot_losses = True, save_path = path1,
            )
        
        
        torch.save(discriminative.state_dict(), path1+'/state_dict.pt')
        pretrained_fig.savefig(path1+'/_pretrain_loss_curves.jpeg')
        learned_fig.savefig(path1+'/pretrain_learned_mixtures.jpeg')
        trained_fig.savefig(path1+'/_train_loss_curves.jpeg')
        plt.close(pretrained_fig)
        plt.close(learned_fig)
        plt.close(trained_fig)

    
    else:
        path1 = root + '/P_X__Y_H'
        discriminative.load_state_dict(torch.load(path1+'/state_dict.pt'))
    

    print('\ncreating visualizations for P_X__Y_H .....')
    stime = time.time()
    HPD_fig, HPD_vid, _, _ = probabilistic_model_HPD(
        model = discriminative, model_name = 'P(X|Y,H)', alpha = 0.05, 
        input_data = test_input[:], label_data = test_label[:], 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 100, plotting = True, video = True,
        )
    LL_fig, LL_vid, _ = probabilistic_model_likelihood(
        model = discriminative, model_name = 'P(X|Y,H)', 
        input_data = test_input[:], label_data = test_label[:], 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 100, plotting = True, video = True,
        )
    print(time.time() - stime)
    
    HPD_fig.savefig(path1+'/test_HPD.jpeg')
    LL_fig.savefig(path1+'/test_likelihood1.jpeg')
    print('  figures saved...')

    HPD_vid.save(path1+'/test_HPD.mp4', writer = 'ffmpeg', dpi = 200)
    LL_vid.save(path1+'/test_likelihood.mp4', writer = 'ffmpeg', dpi = 200)
    print('  videos saved...')
    




    
    
    
    
    




