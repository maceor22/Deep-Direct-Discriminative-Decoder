from copy import deepcopy
from utils import *
from maze_utils import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time as time


# custom Dataset class used during model training
class Data(Dataset):
    
    def __init__(self, inputs, labels):
        self.x = inputs
        self.y = labels
        
    def __len__(self):
        return self.x.size(dim = 0)

    def __getitem__(self, index):
        xi = self.x[index,:]
        yi = self.y[index]
        return xi, yi
    

# MLP generative model that approximates joint probability of XY given input
class OneModelStateProcessNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, feature_dim, history_length):
        # hidden_layer_sizes: tuple or list containing size of hidden layers
        # feature_dim: number of features in data being passed to model
        # history_length: length of the input sequence dimension
        
        super(OneModelStateProcessNN, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, feature_dim*history_length)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
            
        self.lin = nn.Sequential(*layers)
        self.final = nn.Linear(layer_sizes[-1], 4)
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x))
        
    # method for producing generative output and optionally sampling
    def predict(self, pos, return_sample = True):
        # pos: data input
        # return_sample: boolean indicating whether to return a sample from output distribution
        
        out = self.forward(pos)
                
        if return_sample:
            # build covariance matrix/tensor
            covar = torch.zeros((out.size(0), 2, 2))
            covar[:,0,0] = torch.abs(out[:,2])
            covar[:,1,1] = torch.abs(out[:,3])
            # sample from multivariate normal distribution; assumes X-Y independence
            out[:,:2] += MultivariateNormal(torch.zeros((out.size(0),2)), covar).sample()
            # return sample
            return out[:,:2]
        else:
            # return prediction mean and prediction variance
            return out[:,:2], torch.abs(out[:,2:])
        
    
# training scheme for one-model framework using Gaussian negative log-likelihood
def OneModelTrain(
        model, train_data, valid_data, optimizer,
        epochs = 100, batch_size = 64, plot_losses = False,
        ):
    # model: initialized model to be trained
    # train_data: Data object containing training data
    # valid_data: Data object containing validation data
    # optimizer: optimizer object used during training
    # epochs: number of epochs to run during training
    # batch_size: size of batches used during training
    # plot_losses: boolean indicating whether to produce loss plots
    
    # send model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # create dataloader objects for training and validation
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    
    train_losses = []
    valid_losses = []
    
    stime = time.time()
    
    best_loss = 1e10
    
    # begin training scheme
    print('\ntraining.....\n')
    for epoch in range(1,epochs+1):
        train_loss = 0
        valid_loss = 0
        
        model.train()
        
        # iterate over the training data
        for input_batch, label_batch in train_dataloader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            # produce prediction distributions
            pred_mean, pred_vars = model.predict(input_batch, return_sample = False)
            # compute loss
            loss = 2*nn.GaussianNLLLoss()(pred_mean, label_batch, pred_vars)
            
            # zero gradients
            optimizer.zero_grad()
            # backpropagate loss
            loss.backward()
            # prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm = 2.0)
            # update weights
            optimizer.step()
            # aggregate training loss
            train_loss += loss.item()
        
        #compute mean training loss and save to list
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            # iterate over the validation data
            for input_batch, label_batch in valid_dataloader:
                # produce prediction distributions
                pred_mean, pred_vars = model.predict(input_batch, return_sample = False)
                # compute and aggregate validation loss
                valid_loss += 2*nn.GaussianNLLLoss()(pred_mean, label_batch, pred_vars).item()
        
        # compute mean validation loss and save to list
        valid_loss /= len(valid_dataloader)
        valid_losses.append(valid_loss)
        
        # save model that performs best on validation data
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
        
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
    model.load_state_dict(best_state_dict)
    
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
        plt.ylabel('Deviance [-2*LogLikelihood]')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()
        
    return best_epoch, train_losses, valid_losses



# MLP classifier generating probability that next timestep is in each arm 
class ArmNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, num_arms, input_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # num_arms: number of arms in discrete transform
        # input_dim: input dimension to model
        
        super(ArmNN, self).__init__()
        
        self.num_arms = num_arms
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.num_arms))
        
        self.lin = nn.Sequential(*layers)
        self.final = nn.LogSoftmax(dim = 1)
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x))
    
    # method returning optionally output arm or log probability values for each arm
    def predict(self, x, return_log_probs = False):
        if return_log_probs:
            return self.forward(x)
        else:
            x = self.forward(x)
            x = torch.argmax(x, dim = 1)
            out = torch.zeros((x.size(0),self.num_arms))
            for i in range(x.size(0)):
                out[i,x[i]] = 1
            return out
            

# class for training model using negative log-likelihood
class TrainerNLL(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object to be used during training
        self.optimizer = optimizer
    
    # method for training
    def train(self, model, train_data, valid_data,
              epochs = 100, batch_size = 64, plot_losses = False):
        # model: initialized model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        # epochs: number of epochs to run during training
        # batch_size: size of batches to be used during training
        # plot_losses: boolean indicating whether to produce loss plots
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training and validation
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        
        # begin training scheme
        print('\ntraining.....\n')
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                input_batch = ag.Variable(input_batch.to(device))
                label_batch = ag.Variable(label_batch.to(device))
                
                # obtain log-probability values
                log_prob = model(input_batch)
                # compute loss
                loss = 2*nn.NLLLoss()(log_prob, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_norm_(model.parameters(), max_norm = 2.0)
                # update weights
                self.optimizer.step()
                # aggregate training loss
                train_loss += loss.item()
            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    # obtain log-probability values
                    log_prob = model(input_batch)
                    # compute and aggregate validation loss
                    valid_loss += 2*nn.NLLLoss()(log_prob, label_batch).item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
            
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
        model.load_state_dict(best_state_dict)
        
        # produce loss curves
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
            plt.ylabel('Deviance [-2*LogLikelihood]')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses


# MLP generative model determining length traveled along arm and distance from arm center
class DistNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, input_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(DistNN, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
            
        self.lin = nn.Sequential(*layers)
        self.final = nn.Linear(layer_sizes[-1], 4)
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x))
        
    # method that optionally returns prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        out = self.forward(x)
        
        if return_sample:
            # build covariance matrix/tensor
            covar = torch.zeros((out.size(0), 2, 2))
            covar[:,0,0] = torch.abs(out[:,2])
            covar[:,1,1] = torch.abs(out[:,3])
            
            out[:,:2] += MultivariateNormal(torch.zeros((out.size(0),2)), covar).sample()
            # return sample
            return out[:,:2]
        else:
            # return prediction mean and prediction variance
            return out[:,:2], torch.abs(out[:,2:])


# MLP generative model producing X or Y prediction distribution of next timestep
class PositionNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, input_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(PositionNN, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        self.lin = nn.Sequential(*layers)
        self.final = nn.Linear(layer_sizes[-1], 2)
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x))
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        out = self.forward(x)
        
        if return_sample:            
            out[:,0] += Normal(torch.zeros((out.size(0),)), torch.abs(out[:,1])).sample()
            # return sample
            return out[:,0]
        else:
            # return prediction mean and prediction variance
            return out[:,0], torch.abs(out[:,1])


# LSTM generative model producing X-Y prediction distribution of next timestep
class PositionRNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, num_layers, history_length, feature_dim, out_dim = 2):
        # hidden_layer_sizes: integer dictating size of hidden LSTM layers
        # num_layers: integer dictating number of LSTM layers
        # history_length: length of input sequence dimension
        # feature_dim: number of features in input
        # out_dim: size of output dimension; either 2 or 4
        
        super(PositionRNN, self).__init__()
        
        self.out_dim = out_dim
        
        self.lstm = nn.LSTM(
            feature_dim, hidden_layer_sizes, 
            num_layers = num_layers, batch_first = True)
        
        self.final = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_layer_sizes*history_length, out_dim))
        
    # forward call
    def forward(self, x):
        return self.final(self.lstm(x)[0]).squeeze()
    
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        out = self.forward(x)
        
        if self.out_dim == 2:
            if return_sample:
                out[:,0] += Normal(torch.zeros((out.size(0),)), torch.abs(out[:,1])).sample()
                # return sample
                return out[:,0]
            else:
                # return prediction mean and prediction variance
                return out[:,0], torch.abs(out[:,1])
            
        elif self.out_dim == 4:
            if return_sample:
                out[:,:2] += Normal(torch.zeros((out.size(0),2)), torch.abs(out[:,2:])).sample()
                # return sample
                return out[:,:2]
            else:
                # return prediction mean and prediction variance
                return out[:,:2], torch.abs(out[:,2:])


# class for training model using Gaussian negative log-likelihood
class TrainerGaussNLL(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
    
    # method for training
    def train(self, model, train_data, valid_data, 
              epochs = 100, batch_size = 64, plot_losses = False):
        # model: initialized model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        
        print('\ntraining.....\n')
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                input_batch = ag.Variable(input_batch.to(device))
                label_batch = ag.Variable(label_batch.to(device))
                
                # produce prediction distributions
                pred_mean, pred_vars = model.predict(input_batch, return_sample = False)
                # compute loss
                loss = 2*nn.GaussianNLLLoss()(pred_mean, label_batch, pred_vars)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_norm_(model.parameters(), max_norm = 2.0)
                # update weights
                self.optimizer.step()
                # aggregate training loss
                train_loss += loss.item()
            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    # produce prediction distributions
                    pred_mean, pred_vars = model.predict(input_batch, return_sample = False)
                    # compute and aggregate validation loss 
                    valid_loss += 2*nn.GaussianNLLLoss()(pred_mean, label_batch, pred_vars).item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
            
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
        model.load_state_dict(best_state_dict)
        
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
            plt.ylabel('Deviance [-2*LogLikelihood]')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses


# wrapper class for two-model framework
class TwoModelWrapper(object):
    
    def __init__(
            self, modelA, modelB, 
            modelA_multi_output = True,
            modelB_multi_output = True,
            transform = None
            ):
        # modelA: first model in two model framework
        # modelB: second model in two model framework
        # modelA_multi_output: boolean indicating whether modelA outputs mean and variance
        # modelB_multi_output: boolean indicating whether modelB outputs mean and variance
        # transform: fitted transform object; optionally passed
        
        self.modelA = modelA
        self.modelB = modelB
        self.modelA_multi_output = modelA_multi_output
        self.modelB_multi_output = modelB_multi_output
        self.transform = transform
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, untransform_data = False):
        # x: input
        # return_sample: boolean indicating whether to return sample
        # untransform_data: boolean indicating whether to untransform 
        
        if self.modelA_multi_output and self.modelB_multi_output:
            pred_meanA, pred_varsA = self.modelA.predict(x, return_sample = False)
            
            if pred_meanA.dim() != 2:
                xB = torch.cat([x, pred_meanA.unsqueeze(1), pred_varsA.unsqueeze(1)], dim = 1)
            else:
                xB = torch.cat([x, pred_meanA, pred_varsA], dim = 1)
                
            if return_sample:
                pred_meanA = self.modelA.predict(x, return_sample = True)
                pred_meanB = self.modelB.predict(xB, return_sample = True)
                
            else:
                pred_meanB, pred_varsB = self.modelB.predict(xB, return_sample = False)
                
                if pred_meanA.dim() != 2 and pred_meanB.dim() != 2:
                    pred_vars = torch.cat([pred_varsA.unsqueeze(1), pred_varsB.unsqueeze(1)], dim = 1)
                else:
                    pred_vars = torch.cat([pred_varsA, pred_varsB], dim = 1)
            
            if pred_meanA.dim() != 2 and pred_meanB.dim() != 2:
                pred_data = torch.cat([pred_meanA.unsqueeze(1), pred_meanB.unsqueeze(1)], dim = 1)
            else:
                pred_data = torch.cat([pred_meanA, pred_meanB], dim = 1)
        
        elif not self.modelA_multi_output and self.modelB_multi_output:
            predA = self.modelA.predict(x, return_log_probs = False)
            predA_probs = torch.exp(self.modelA.predict(x, return_log_probs = True))
            
            xB = torch.cat([x, predA_probs], dim = 1)
            if return_sample:
                pred_meanB = self.modelB.predict(xB, return_sample = True)
                
            else:
                pred_meanB, pred_vars = self.modelB.predict(xB, return_sample = False)
            
            if pred_meanB.dim() != 2:
                pred_data = torch.cat([predA, pred_meanB.unsqueeze(1)], dim = 1)
            else:
                pred_data = torch.cat([predA, pred_meanB], dim = 1)
            
        elif self.modelA_multi_output and not self.modelB_multi_output:
            pred_meanA, pred_vars = self.modelA.predict(x, return_sample = False)
            
            xB = torch.cat([x, pred_meanA, pred_vars], dim = 1)
            
            predB = self.modelB.predict(xB)
            if return_sample:
                pred_meanA = self.modelA.predict(x, return_sample = True)
                                
            pred_data = torch.cat([predB, pred_meanA], dim = 1)
        
        
        if return_sample and untransform_data:
            return self.transform.untransform(pred_data.unsqueeze(dim = 1)).squeeze()
        
        elif not return_sample and untransform_data:
            untf_pred_data, untf_pred_vars = self.transform.untransform(
                pred_data.unsqueeze(dim = 1), pred_vars)
            return untf_pred_data.squeeze(), untf_pred_vars
        
        elif not return_sample and not untransform_data:
            return pred_data, pred_vars
        
        else:
            return pred_data
        

# training scheme for two-model framework
def TwoModelTrain(
        modelA, modelB, 
        trainerA, trainerB,
        train_input, valid_input, 
        train_labelA, valid_labelA,
        train_labelB, valid_labelB,
        modelA_multi_output = True,
        epochs = (100, 100), batch_size = 64, plot_losses = False,
        ):
    # modelA: first model in two-model framework
    # modelB: second model in two-model framework
    # trainerA: trainer object for training modelA
    # trainerB: trainer object for training modelB
    # train_input: input data for training
    # valid_input: input data for validation
    # train_labelA: training labels for modelA
    # valid_labelA: validation labels for modelA
    # train_labelB: training labels for modelB
    # valid_labelB: validation labels for modelB
    # modelA_multi_output: boolean indicating whether modelA produces mean and variance
    # epochs: list or tuple of length 2 dictating how many epochs to use in training modelA and modelB
    # batch_size: size of batches to be used during training
    # plot_losses: boolean indicating whether to produce loss plots
    
    # create training and validation Data objects for modelA 
    train_dataA = Data(train_input, train_labelA)
    valid_dataA = Data(valid_input, valid_labelA)
    
    # run training scheme for modelA
    trainerA.train(
        modelA, train_dataA, valid_dataA,
        epochs = epochs[0], batch_size = batch_size, plot_losses = plot_losses,
        )
    
    # get modelA training predictions and validation predictions
    
    if modelA_multi_output:
        tr_pred_mean, tr_pred_vars = modelA.predict(train_input, return_sample = False)
        va_pred_mean, va_pred_vars = modelA.predict(valid_input, return_sample = False)
        
        if tr_pred_mean.dim() != 2:
            tr_pred = torch.cat([tr_pred_mean.unsqueeze(1), tr_pred_vars.unsqueeze(1)], dim = 1)
            va_pred = torch.cat([va_pred_mean.unsqueeze(1), va_pred_vars.unsqueeze(1)], dim = 1)
        else:
            tr_pred = torch.cat([tr_pred_mean, tr_pred_vars], dim = 1)
            va_pred = torch.cat([va_pred_mean, va_pred_vars], dim = 1)
        
    else:
        tr_pred = torch.exp(modelA.predict(train_input, return_log_probs = True))
        va_pred = torch.exp(modelA.predict(valid_input, return_log_probs = True))
    
    # concatenate training and validation inputs with training and validation predictions
    train_inputB = torch.cat([train_input, tr_pred], dim = 1)
    valid_inputB = torch.cat([valid_input, va_pred], dim = 1)
    
    # create training and validation Data objects for modelB
    train_dataB = Data(train_inputB, train_labelB)
    valid_dataB = Data(valid_inputB, valid_labelB)
    
    # run training scheme for modelB
    trainerB.train(
        modelB, train_dataB, valid_dataB,
        epochs = epochs[1], batch_size = batch_size, plot_losses = plot_losses,
        )




    


