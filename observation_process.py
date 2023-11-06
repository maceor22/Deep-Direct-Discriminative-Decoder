import os
from copy import deepcopy
import torch
import numpy as np
from torch import nn
from torch import autograd as ag
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from misc_plotting import *
from maze_utils import Maze, Data, RangeNormalize
from models import *
from trainers import *
from data_generation import generate_dataset, balance_dataset
import time
import datetime
from memory_profiler import profile

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
    
class P_X__H(nn.Module):

    def __init__(self, P_Y__H, P_X__Y_H):

        super(P_X__H, self).__init__()

        self.P_Y__H = P_Y__H
        self.P_X__Y_H = P_X__Y_H
    
    #@profile
    def prob(self, h, x, n_samples = 10):
        probs = torch.zeros((x.size(0),), device = x.device)

        y_distribution = Binomial(
            total_count = self.P_Y__H.n_trials, logits = self.P_Y__H(h))

        for _ in range(n_samples):
            y = y_distribution.sample().unsqueeze(1)
            probs += self.P_X__Y_H.prob(torch.cat([h, y], dim = 1), x)
            del y

        return probs


class ObservationProcess(nn.Module):

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
        return probs



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
        mixture_network, covariance_type, 
        train_input, train_label, valid_input, valid_label,
        epochs = 100, batch_size = 256, plot_losses = False,
        suppress_prints = False,
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
                input_batch = valid_input[ix[lower:upper]].to(device)
                
                #pi_target = valid_z[ix[lower:upper]]
                pi_target = tr_pi_target[ix[lower:upper]].to(device)
                mu_target = va_mu_target[ix[lower:upper]].to(device)
                var_target = va_var_target[ix[lower:upper]].to(device)
                
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



def train_count_process(
        count_process, train_data, valid_data, log_target = True,
        epochs = 100, batch_size = 256, plot_losses = False,
        suppress_prints = False,
        ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    count_process = count_process.to(device)

    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    
    train_losses = []
    valid_losses = []
    
    stime = time.time()
    
    best_loss = 1e10

    KLDiv = nn.KLDivLoss(reduction = 'batchmean', log_target = log_target)
    optimizer = torch.optim.SGD(count_process.parameters(), lr = 1e-3)
    
    if not suppress_prints:
        print('\ntraining.....\n')
    for epoch in range(1,epochs+1):        
        train_loss = 0
        valid_loss = 0
        
        count_process.train()
        # iterate over the training data
        for input_batch, label_batch in train_dataloader:
            input_batch = ag.Variable(input_batch.to(device))
            label_batch = ag.Variable(label_batch.to(device))
            
            # produce prediction distributions
            pred = count_process.predict(input_batch)
            loss = KLDiv(pred, label_batch)

            # zero gradients
            optimizer.zero_grad()
            # backpropagate loss
            loss.backward()
            # prevent exploding gradients
            clip_grad_value_(count_process.parameters(), clip_value = 2)
            # update weights
            optimizer.step()
            # aggregate training loss
            train_loss += loss.item()
        
        # compute mean training loss and save to list
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        count_process.eval()
        with torch.no_grad():
            # iterate over validation data
            for input_batch, label_batch in valid_dataloader:
                # produce prediction distributions
                pred = count_process.predict(input_batch.to(device))
                loss = KLDiv(pred, label_batch.to(device))
                # compute and aggregate validation loss 
                valid_loss += loss.item()
        
        # compute mean validation loss and save to list
        valid_loss /= len(valid_dataloader)
        valid_losses.append(valid_loss)
        
        # save model that performs best on validation data
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state_dict = deepcopy(count_process.state_dict())
        
        if not suppress_prints:
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
    count_process.load_state_dict(best_state_dict)
    
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
        plt.ylabel('KL')
        plt.title('Loss Curves')
        plt.legend()

        return best_epoch, train_loss, valid_losses, fig
        
    return best_epoch, train_losses, valid_losses



if __name__ == '__main__':
    
    """ plot_density_mixture_metrics(
        root = 'ObservationModels/history eval/_historyEval_2layerGMMLP_8mix_hid12_2023-9-19_21-14-46',
    ) """

    """ density_mixture_history_evaluation(
        num_mixtures = 3, hidden_layers = [48,48],
        test_size = 1000, num_folds = 5,
        min_history = 0, max_history = 17, bin_size = 1, step = 4,
        pretrain_epochs = 500, training_epochs = 5000,
        root = 'ObservationModels/history eval/GMMLP_3mix_hid[48, 48]_2023-10-5_11-29-37',
        HPD_alpha = 0.05, include_velocity = False, dirvel = False, plot = True,   
    ) """

    shl = 20
    bs = 1
    
    data = generate_dataset(
        rat_name = 'Bon', 
        input_history_length = shl, spike_bin_size = bs, label_history_length = 0, 
        include_velocity = True, dirvel = False, dmin = 0.5, dmax = 20,
        grid_resolution = 0.1, balance_resolution = 0.1, 
        threshold = 100, presence = False, p_range = 2,
        )
    
    bal_train_input, bal_train_label = data['train_spikes'], data['train_labels']
    bal_valid_input, bal_valid_label = data['valid_spikes'], data['valid_labels']
    
    raw_test_input, test_label = data['test_spikes'], data['test_labels']
    
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    
    print('\n ', bal_train_input.size(), bal_train_label.size())
    
    """ spikeNorm = RangeNormalize(dim = bal_train_input.size(2), norm_mode = 'auto')
    spikeNorm.fit(
        range_min = [0 for _ in range(bal_train_input.size(2))], 
        range_max = [10 for _ in range(bal_train_input.size(2))],
        )
    
    train_input = spikeNorm.transform(bal_train_input)
    valid_input = spikeNorm.transform(bal_valid_input)
    test_input = spikeNorm.transform(raw_test_input) """

    print(bal_train_input.max(0)[0].max(0)[0].max(0)[0])
    train_input = torch.log(bal_train_input + 1)
    valid_input = torch.log(bal_valid_input + 1)
    test_input = torch.log(raw_test_input + 1)

    posNorm = RangeNormalize(dim = 2, norm_mode = [0,0,])
    posNorm.fit(
        range_min = (xmin, ymin,), range_max = (xmax, ymax,),
        )
    
    train_label = posNorm.transform(bal_train_label[:,:2]).float()
    valid_label = posNorm.transform(bal_valid_label[:,:2]).float()

    print(train_input.size(), valid_input.size(), test_input.size())

    
    #root = None
    #root = 'ObservationModels/trained/MultivariateNormalMixtureMLP_2023-10-2_1-7-53'
    root = 'ObservationModels/trained/MVN-Mix-MLP_Binomial-MLP_2023-10-14_20-2-12'

    
    name = 'MVN-Mix-MLP_Binomial-MLP'
    P_x__y_h = MultivariateNormalMixtureMLP(
        hidden_layer_sizes = [24,24],
        num_mixtures = 5,
        input_dim = train_input.size(1) * train_input.size(2),
        latent_dim = 2,
    )
    P_y__h = BinomialMLP(
        hidden_layer_sizes = [24,24],
        input_dim = (train_input.size(1) - 1) * train_input.size(2),
        latent_dim = train_input.size(2),
        n_trials = 10, log_target = True,
    )
    covar_type = 'full'
    

    if root == None:

        print('\npre-training P(X|Y,H) .....\n')
        _, _, _, pretrained_fig1 = pretrain_density_mixture_network(
            mixture_network = P_x__y_h, 
            covariance_type = covar_type, 
            train_input = train_input, train_label = train_label, 
            valid_input = valid_input, valid_label = valid_label,
            epochs = 10, plot_losses = True,
            )

        trainer1 = TrainerMLE(
            optimizer = torch.optim.SGD(P_x__y_h.parameters(), lr = 1e-3))
        
        _, _, _, trained_fig1 = trainer1.train(
            model = P_x__y_h, 
            train_data = Data(train_input, train_label), 
            valid_data = Data(valid_input, valid_label),
            epochs = 10, batch_size = 256, plot_losses = True,
            )
        

        print('\n\ntraining P(Y|H) .....')
        trainer2 = TrainerMLE(
            optimizer = torch.optim.SGD(P_y__h.parameters(), lr = 1e-3))
        
        _, _, _, trained_fig2 = trainer2.train(
            model = P_y__h, 
            train_data = Data(train_input[:,:-1,:], train_input[:,-1,:]), 
            valid_data = Data(valid_input[:,:-1,:], valid_input[:,-1,:]),
            epochs = 1000, batch_size = 256, plot_losses = True,
            )
        
        now = datetime.datetime.now()
        root = f'ObservationModels/trained/{name}_'
        root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
        path1 = root + '/P_X__Y_H'
        path2 = root + '/P_Y__H'
        path3 = root + '/P_X__H'
        path4 = root + '/Ratio'
        os.mkdir(root)
        os.mkdir(path1)
        os.mkdir(path2)
        os.mkdir(path3)
        os.mkdir(path4)
        
        torch.save(P_x__y_h.state_dict(), path1+'/state_dict.pt')
        pretrained_fig1.savefig(path1+'/_pretrain_loss_curves.jpeg')
        trained_fig1.savefig(path1+'/_train_loss_curves.jpeg')
        plt.close(pretrained_fig1)
        plt.close(trained_fig1)

        torch.save(P_y__h.state_dict(), path2+'/state_dict.pt')
        trained_fig2.savefig(path2+'/_train_loss_curves.jpeg')
        plt.close(trained_fig2)
    
    else:
        path1 = root + '/P_X__Y_H'
        path2 = root + '/P_Y__H'
        path3 = root + '/P_X__H'
        path4 = root + '/Ratio'

        P_x__y_h.load_state_dict(torch.load(path1+'/state_dict.pt'))
        P_y__h.load_state_dict(torch.load(path2+'/state_dict.pt'))
        P_x__h = P_X__H(P_Y__H = P_y__h, P_X__Y_H = P_x__y_h)
        observation_process = ObservationProcess(P_x__y_h, P_x__h)


    print('\ncreating visualizations for P_X__Y_H .....')
    """ HPD_fig1, HPD_vid1, _, _ = density_mixture_HPD(
        mixture_model = P_x__y_h, covariance_type = covar_type,
        input_data = test_input, label_data = test_label, posNorm = posNorm, 
        grid_dim = 100, untransform_label = False, alpha = 0.05, 
        plotting = True, video = True,
        ) """
    """ stime = time.time()
    LL_fig1, LL_vid1, _ = probabilistic_model_likelihood(
        model = P_x__y_h, model_name = 'P(X|Y,H)', 
        input_data = test_input, label_data = test_label, 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 100, plotting = True, video = True,
        )
    print(time.time() - stime)
    
    #HPD_fig1.savefig(path1+'/test_HPD.jpeg')
    LL_fig1.savefig(path1+'/test_likelihood.jpeg')
    print('  figures saved...')

    #HPD_vid1.save(path1+'/test_HPD.mp4', writer = 'ffmpeg', dpi = 200)
    LL_vid1.save(path1+'/test_likelihood.mp4', writer = 'ffmpeg', dpi = 200)
    print('  videos saved...') """
    

    print('\ncreating visualizations for P_X__H')
    """ stime = time.time()
    HPD_fig3, HPD_vid3, _, _ = density_mixture_HPD(
        mixture_model = P_x__h, covariance_type = covar_type,
        input_data = test_input[:,:-1,:], label_data = test_label, posNorm = posNorm, 
        grid_dim = 100, untransform_label = False, alpha = 0.05, 
        plotting = True, video = True,
        )
    print(time.time() - stime)
    plt.show() """

    """ stime = time.time()
    LL_fig3, LL_vid3, _ = probabilistic_model_likelihood(
        model = P_x__h, model_name = 'P(X|H)', 
        input_data = test_input[:,:-1,:], label_data = test_label, 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 50, plotting = True, video = True,
        )
    print(time.time() - stime)
    
    #HPD_fig3.savefig(path3+'/test_HPD.jpeg')
    LL_fig3.savefig(path3+'/test_likelihood.jpeg')
    print('  figures saved...')

    #HPD_vid3.save(path3+'/test_HPD.mp4', writer = 'ffmpeg', dpi = 200)
    LL_vid3.save(path3+'/test_likelihood.mp4', writer = 'ffmpeg', dpi = 200)
    print('  videos saved...') """


    print('\ncreating visualizations for P(X|Y,H)^2 / P(X|H)')
    """stime = time.time()
    HPD_fig2, HPD_vid2, _, _ = density_mixture_HPD(
        mixture_model = P_x__h, covariance_type = covar_type,
        input_data = test_input[:,:-1,:], label_data = test_label, posNorm = posNorm, 
        grid_dim = 100, untransform_label = False, alpha = 0.05, 
        plotting = True, video = True,
        )
    print(time.time() - stime)
    plt.show() """

    stime = time.time()
    LL_fig4, LL_vid4, _ = probabilistic_model_likelihood(
        model = observation_process, model_name = 'P(X|Y,H)^2 / P(X|H)', 
        input_data = test_input, label_data = test_label, 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 50, plotting = True, video = True,
        )
    print(time.time() - stime)
    
    #HPD_fig2.savefig(root+'/P_X__H/test_HPD.jpeg')
    LL_fig4.savefig(path4+'/test_likelihood.jpeg')
    print('  figures saved...')

    #HPD_vid2.save(root+'/P_X__H/test_HPD.mp4', writer = 'ffmpeg', dpi = 200)
    LL_vid4.save(path4+'/test_likelihood.mp4', writer = 'ffmpeg', dpi = 200)
    print('  videos saved...')
    
    
    
    
    




