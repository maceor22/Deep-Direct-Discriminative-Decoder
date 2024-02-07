from copy import deepcopy
from maze_utils import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import time as time
import os
import json



class TrainerMLE(object):
    
    def __init__(self, optimizer, suppress_prints = False, print_every = 10):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every
    
    # method for training
    def train(self, model, train_data, valid_data, grad_clip_value = 5,
              epochs = 100, batch_size = 64, shuffle = True, 
              plot_losses = False, save_path = None,
              ):
        # model: initialized model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = shuffle)
        valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = shuffle)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        best_state_dict = deepcopy(model.state_dict())
        
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(device))
                    label_batch = ag.Variable(label_batch.to(device))
                    
                    # minimize negative log likelihood
                    nll = -1 * model.log_prob(input_batch, label_batch).mean(dim = 0)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    nll.backward()
                    # prevent exploding gradients
                    clip_grad_value_(model.parameters(), clip_value = grad_clip_value)
                    # update weights
                    self.optimizer.step()
                    # aggregate training loss
                    train_loss += nll.item()
                
                except:
                    # if NaNs encountered in gradients
                    print('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        nll = -1 * model.log_prob(
                            input_batch.to(device), label_batch.to(device)).mean(dim = 0)
                        # compute and aggregate validation loss 
                        valid_loss += nll.item()
                    
                    except:
                        print('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())

                if save_path is not None and os.path.exists(save_path):
                    torch.save(best_state_dict, save_path+'/state_dict.pt')

                    with open(save_path+'/losses.json', 'w') as f:
                        json.dump(
                            {'train losses' : train_losses, 'valid losses': valid_losses}, f
                        )

            
            if not self.suppress:
                # printing
                if epoch % self.print_every == 0:
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
    


class TrainerMLE1(object):
    
    def __init__(self, optimizer, suppress_prints = False):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
        self.suppress = suppress_prints
        print(torch.cuda.device_count(), 'GPUs available')
        self.distributed = torch.cuda.device_count() > 1
        print('distributed bool: ', self.distributed)

    
    def train(
            self, model, train_data, valid_data, grad_clip_value = 5,
            epochs = 100, batch_size = 256, shuffle = True, plot_losses = False,
            ):
        if self.distributed:
            world_size = torch.cuda.device_count()
            mp.spawn(
                self.multiprocess,
                args = (world_size, model, train_data, valid_data,
                        grad_clip_value, epochs, batch_size,
                        shuffle, plot_losses,
                        ),
                nprocs = world_size,
                )
        
        else:
            rank = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.single_process(
                rank, model, train_data, valid_data, grad_clip_value, epochs,
                batch_size, shuffle, plot_losses,
            )


    def DDP_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        init_process_group(backend = 'nccl', rank = rank, world_size = world_size)
        torch.cuda.set_device(rank)


    def multiprocess(
            self, rank, world_size, model, train_data, valid_data, 
            grad_clip_value, epochs, batch_size, shuffle, plot_losses,
            ):
        self.DDP_setup(rank, world_size)

        self.single_process(
            rank, model, train_data, valid_data, grad_clip_value, epochs,
            batch_size, shuffle, plot_losses,
            )
        
        destroy_process_group()

    
    # method for training
    def single_process(
            self, rank, model, train_data, valid_data, grad_clip_value,
            epochs, batch_size, shuffle, plot_losses,
            ):
        # model: initialized model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        
        # send model to device
        model = model.to(rank)

        if self.distributed:
            model = DDP(model, device_ids = [rank])
        
        # create dataloader objects for training data and validation data
        if self.distributed:
            train_dataloader = DataLoader(
                train_data, batch_size = batch_size, shuffle = False,
                sampler = DistributedSampler(train_data),
                )
            valid_dataloader = DataLoader(
                valid_data, batch_size = batch_size, shuffle = False,
                sampler = DistributedSampler(valid_data),
                )
        else:
            train_dataloader = DataLoader(
                train_data, batch_size = batch_size, shuffle = shuffle,
                )
            valid_dataloader = DataLoader(
                valid_data, batch_size = batch_size, shuffle = shuffle,
                )
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e10
        if self.distributed:
            best_state_dict = deepcopy(model.module.state_dict())
        else:
            best_state_dict = deepcopy(model.state_dict())
        
        for epoch in range(1,epochs+1):
            if self.distributed and shuffle:
                train_dataloader.sampler.set_epoch(epoch)
                valid_dataloader.sampler.set_epoch(epoch)

            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(rank))
                    label_batch = ag.Variable(label_batch.to(rank))
                    
                    # minimize negative log likelihood
                    nll = -1 * model.log_prob(input_batch, label_batch).mean(dim = 0)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    nll.backward()
                    # prevent exploding gradients
                    clip_grad_value_(model.parameters(), clip_value = grad_clip_value)
                    # update weights
                    self.optimizer.step()
                    # aggregate training loss
                    train_loss += nll.item()
                
                except:
                    # if NaNs encountered in gradients
                    print('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        nll = -1 * model.log_prob(
                            input_batch.to(rank), label_batch.to(rank)).mean(dim = 0)
                        # compute and aggregate validation loss 
                        valid_loss += nll.item()
                    
                    except:
                        print('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                if self.distributed:
                    best_state_dict = deepcopy(model.module.state_dict())
                else:
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






