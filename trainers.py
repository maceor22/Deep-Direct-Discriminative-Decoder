from copy import deepcopy
from maze_utils import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
import time as time
            

# class for training model using negative log-likelihood
class TrainerNLL(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object to be used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
    
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
        if not self.suppress:
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
            
            if not self.suppress:
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



# class for training model using Gaussian negative log-likelihood
class TrainerGaussianNLL(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
    
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
        
        if not self.suppress:
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
                clip_grad_value_(model.parameters(), clip_value = 5.0)
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
            
            if not self.suppress:
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



class TrainerMultivariateNormalNLL(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
    
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
        
        if not self.suppress:
            print('\ntraining.....\n')
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                input_batch = ag.Variable(input_batch.to(device))
                label_batch = ag.Variable(label_batch.float().to(device))
                
                # produce prediction distributions
                pred_mean, pred_scale_tril = model.predict(
                    input_batch, return_sample = False, scale_tril = True,
                    )
                # compute loss
                mvn = MultivariateNormal(loc = pred_mean, scale_tril = pred_scale_tril)
                loss = -mvn.log_prob(label_batch).mean(dim = 0)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(model.parameters(), clip_value = 5.0)
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
                    pred_mean, pred_scale_tril = model.predict(
                        input_batch, return_sample = False, scale_tril = True,
                        )
                    # compute and aggregate validation loss 
                    valid_loss += -MultivariateNormal(
                        loc = pred_mean, scale_tril = pred_scale_tril,
                        ).log_prob(label_batch.float()).mean(dim = 0).item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
            
            if not self.suppress:
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
    


class TrainerExpectationMaximization(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
    
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
        
        if not self.suppress:
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
                loss = model.expect(input_batch, label_batch)
                                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(model.parameters(), clip_value = 2)
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
                    loss = model.expect(input_batch, label_batch)
                    # compute and aggregate validation loss 
                    valid_loss += loss.item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
            
            if not self.suppress:
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
            plt.ylabel('Negative Mean Log Expectation')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses


class TrainerExpectationMaximization1(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
    
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
        
        if not self.suppress:
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
                loss, gamma = model.expect(input_batch, label_batch)
                
                model.maximize(gamma, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(model.parameters(), clip_value = 1.0)
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
                    expectation = model.expect(input_batch, label_batch)
                    # compute and aggregate validation loss 
                    valid_loss += (-1 * torch.log(expectation).mean(dim = 0)).item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
            
            if not self.suppress:
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
            plt.ylabel('Negative Mean Log Expectation')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses


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
            tr_pred = torch.cat([tr_pred_mean, tr_pred_vars.flatten(1,-1)], dim = 1)
            va_pred = torch.cat([va_pred_mean, va_pred_vars.flatten(1,-1)], dim = 1)
        
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






