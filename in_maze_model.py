import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import time
from copy import deepcopy
from maze_utils import Maze, RangeNormalize
from state_process_models import Data
from data_generation import generate_out_maze_data, generate_trajectory_history


# method for generating history of in-maze probabilities
def generate_inside_maze_prob_history(data, inside_maze_model, flatten = False):
    # data: trajectory data
    # inside_maze_model: in-maze classifier
    # flatten: boolean indicating whether whether data should be flattened 
    #   before passed to in-maze classifier
    
    # infer history length from data
    hl = data.size(1)
    
    # generate probability values from in_maze_model
    if flatten:
        probs = inside_maze_model(data.flatten(1,-1))
    else:
        probs = inside_maze_model(data)
    
    prob_hist = []
    for i in range(hl, probs.size(0)):
        # generate probability history for each timestep
        prob_hist.append(probs[i-hl:i].unsqueeze(0))
        
    # concatenate probability histories for all time steps
    prob_hist = torch.cat(prob_hist, dim = 0).unsqueeze(2)
    
    # concatenate probability histories with trajectory histories
    return torch.cat([data[hl:], prob_hist], dim = 2)
    

# MLP classifier model producing probability that a given trajectory is inside the maze
class InMazeModelNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, feature_dim, history_length):
        # hidden_layer_sizes: list or tuple containing sizes of hidden layers
        # feature_dim: number of features in input
        # history_length: length of input sequence dimension
        
        super(InMazeModelNN, self).__init__()
                
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, feature_dim*history_length)
        
        # build hidden layers with ReLU activation inbetween
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], 1))
        
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(*layers)
        self.final = nn.Sigmoid()
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(self.flatten(x))).squeeze()
    
    # method for outputting binary class (0 for out-maze, 1 for in-maze)
    def predict(self, x):
        x = self.forward(x)
        out = torch.where(x > 0.5, 1, 0)
        return out


# LSTM classifier model producing probability that a given trajectory is inside the maze
class InMazeModelRNN(nn.Module):
    
    def __init__(self, hidden_layer_sizes, num_layers, feature_dim, history_length):
        # hidden_layer_sizes: integer dictating LSTM hidden layer sizes
        # num_layers: number of LSTM layers
        # feature_dim: number of features in input
        # history_length: length of input sequence dimension
        
        super(InMazeModelRNN, self).__init__()
        
        self.lstm = nn.LSTM(
            feature_dim, hidden_layer_sizes, 
            num_layers = num_layers, batch_first = True)
        
        self.final = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden_layer_sizes*history_length, 1), nn.Sigmoid())
        
    # forward call
    def forward(self, x):
        return self.final(self.lstm(x)[0]).squeeze()
    
    # method for outputting binary class (0 for out-maze, 1 for in-maze)
    def predict(self, x):
        x = self.forward(x)
        out = torch.where(x > 0.5, 1, 0)
        return out
        

# class for training classifier model using binary cross entropy
class TrainerBCE(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
    
    # method for training classifer model
    def train(self, model, train_data, valid_data,
              epochs = 100, batch_size = 64, plot_losses = False):
        # model: classifier model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        # epochs: number of epochs to run during training
        # batch_size: size of batches to use during training
        # plot_losses: boolean indicating whether to produce loss plots
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create training and validation dataloader objects
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
                
                # generate predicted probability values
                input_prob = model(input_batch)
                # compute loss
                loss = nn.BCELoss()(input_prob, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_norm_(model.parameters(), max_norm = 2.0)
                # update weights
                self.optimizer.step()
                # aggregate training losses
                train_loss += loss.item()
            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    # generate prediction probability values
                    input_prob = model(input_batch)
                    # compute and aggregate validation losses
                    valid_loss += nn.BCELoss()(input_prob, label_batch).item()
            
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
        
        # plotting
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
            plt.ylabel('Binary Cross Entropy Loss')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses
    

# class for training classifier model using BCE
class TrainerBCE1(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
    
    # method for training classifier model
    def train(
            self, model, 
            train_input0, train_input1, valid_input0, valid_input1,
            epochs = 100, batch_size = 64, plot_losses = False
            ):
        # model: model to be trained
        # train_input0: input training data belonging to class 0 (out-maze)
        # train_input1: input training data belonging to class 1 (in-maze)
        # valid_input0: input validation data belonging to class 0 (out-maze)
        # valid_input1: input validation data belonging to class 1 (in-maze)
        # epochs: epochs to run during training
        # batch_size: size of batches to use during training
        # plot_losses: boolean indicating whether to produce loss plots
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        
        tr_label0 = torch.zeros((train_input1.size(0),))
        tr_label1 = torch.ones((train_input1.size(0),))
        train_label = torch.cat([tr_label0, tr_label1], dim = 0)
        
        va_label0 = torch.zeros((valid_input1.size(0),))
        va_label1 = torch.ones((valid_input1.size(0),))
        valid_label = torch.cat([va_label0, va_label1], dim = 0)
        
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 0
            
        print('\ntraining.....\n')
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            # sample from class 0 data for balanced data between class 0 and class 1
            tr_ix0 = torch.randperm(train_input0.size(0))[:train_input1.size(0)]
            tr_input0 = train_input0[tr_ix0]
            
            va_ix0 = torch.randperm(valid_input0.size(0))[:valid_input1.size(0)]
            va_input0 = valid_input0[va_ix0]
            
            train_input = torch.cat([tr_input0, train_input1], dim = 0)
            valid_input = torch.cat([va_input0, valid_input1], dim = 0)
            
            train_dataloader = DataLoader(
                Data(train_input, train_label), batch_size = batch_size, shuffle = True)
            valid_dataloader = DataLoader(
                Data(valid_input, valid_label), batch_size = batch_size, shuffle = True)
            
            
            model.train()
            # iterating over the training data
            for input_batch, label_batch in train_dataloader:
                input_batch = ag.Variable(input_batch.to(device))
                label_batch = ag.Variable(label_batch.to(device))
                
                input_prob = model(input_batch)
                
                loss = nn.BCELoss()(input_prob, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_norm_(model.parameters(), max_norm = 2.0)
                # update weights
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # compute mean training loss and append to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data
                for input_batch, label_batch in valid_dataloader:
                    # generate prediction probability values
                    input_prob = model(input_batch)
                    # compute and aggregate validation losses
                    valid_loss += nn.BCELoss()(input_prob, label_batch).item()
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)
                
            # save model that performs best on validation data
            if valid_loss > best_loss:
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
        
        # plotting
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
            plt.ylabel('Binary Cross Entropy')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses


# class for training classifier model using AUROC
class TrainerAUROC(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
    
    # method for training classifier model
    def train(
            self, model, 
            train_input0, train_input1, valid_input0, valid_input1,
            epochs = 100, batch_size = 64, plot_losses = False
            ):
        # model: model to be trained
        # train_input0: input training data belonging to class 0 (out-maze)
        # train_input1: input training data belonging to class 1 (in-maze)
        # valid_input0: input validation data belonging to class 0 (out-maze)
        # valid_input1: input validation data belonging to class 1 (in-maze)
        # epochs: epochs to run during training
        # batch_size: size of batches to use during training
        # plot_losses: boolean indicating whether to produce loss plots
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create validation labels to be used in AUC calculation
        va_auc_target = np.concatenate((
            np.zeros((valid_input1.size(0),)), np.ones(valid_input1.size(0),)), axis = 0)
        
        # create labels for dlogits in BCE loss
        bce_target = torch.ones((batch_size,))
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 0
            
        print('\ntraining.....\n')
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            # sample from class 0 data for balanced data between class 0 and class 1
            ix0 = torch.randperm(train_input0.size(0))[:train_input1.size(0)]
            # shuffle class 1 input data
            ix1 = torch.randperm(train_input1.size(0))
                        
            model.train()
            
            train_done = False
            
            lower = 0
            upper = batch_size
            num_batches = 0
            
            # iterating over the training data
            while not train_done:
                num_batches += 1
                
                # create input batches for class 0 and class 1 data
                input_batch0 = ag.Variable(
                    train_input0[ix0[lower:upper]].to(device))
                input_batch1 = ag.Variable(
                    train_input1[ix1[lower:upper]].to(device))
                
                # generate probability values for class 0 batch and class 1 batch
                prob0 = model(input_batch0)
                prob1 = model(input_batch1)
                
                # concatenate and convert to numpy
                probs = torch.cat([prob0, prob1], dim = 0).detach().numpy()
                
                # create training labels to be used in AUC calculation
                tr_auc_target = np.concatenate((
                    np.zeros((prob0.size(0),)), np.ones((prob1.size(0),))), axis = 0)
                
                # aggregate training AUC losses
                train_loss += roc_auc_score(tr_auc_target, probs)
                
                # compute logits
                logit0 = torch.log(prob0 / (1 - prob0))
                logit1 = torch.log(prob1 / (1 - prob1))
                
                # compute difference between logits
                dlogit = logit1 - logit0
                
                # compute BCE loss
                loss = nn.BCEWithLogitsLoss()(dlogit, bce_target)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_norm_(model.parameters(), max_norm = 2.0)
                # update weights
                self.optimizer.step()
                
                # next batch                
                lower += batch_size
                upper += batch_size
                
                
                if upper > train_input1.size(0):
                    upper = -1
                if upper == -1:
                    train_done = True
            
            # compute mean training loss and append to list
            train_loss /= num_batches
            train_losses.append(train_loss)
            
            
            model.eval()
            
            # sample from class 0 data for balanced data between class 0 and class 1
            ix0 = torch.randperm(valid_input0.size(0))[:valid_input1.size(0)]
            va_input0 = valid_input0[ix0]
            
            with torch.no_grad():
                # compute probability values for class 0 and class 1, then concatenate
                prob0 = model(va_input0)
                prob1 = model(valid_input1)
                probs = torch.cat([prob0, prob1], dim = 0)
                
                # compute validation loss and append to list
                valid_loss = roc_auc_score(va_auc_target, probs)
                valid_losses.append(valid_loss)
                
            # save model that performs best on validation data
            if valid_loss > best_loss:
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
        
        # plotting
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
            plt.ylabel('AUROC')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses
    

# method for loading out-maze data
def load_out_maze_datasets(
        path, history_length = 1, 
        train_ratio = .85, valid_ratio = .10, 
        transform = None,
        ):
    # path: file path to out-maze data
    # history_length: desired length of trajectory
    # train_ratio: ratio of data used for training
    # valid_ratio: ratio of data used for validation
    # transform: fitted transform object; optionally passed
    
    # load data and generate trajectory history with desired history length
    out_data = torch.load(path)
    input_data = generate_trajectory_history(out_data, history_length)
    
    if transform is not None:
        # apply transform
        input_data = transform.transform(input_data)
    
    # bounds used for train/validation/test split
    bound0 = int(train_ratio * input_data.size(0))
    bound1 = int((train_ratio+valid_ratio) * input_data.size(0))
    
    # split inputs into train/validation/test sets
    tr_input = input_data[:bound0]
    va_input = input_data[bound0:bound1]
    te_input = input_data[bound1:]
    
    # create labels for train/validation/test sets
    tr_label = torch.zeros((tr_input.size(0),))
    va_label = torch.zeros((va_input.size(0),))
    te_label = torch.zeros((te_input.size(0),))
    
    # create train/validation/test Data objects
    train_data = Data(tr_input, tr_label)
    valid_data = Data(va_input, va_label)
    test_data = Data(te_input, te_label)
    
    return train_data, valid_data, test_data




if __name__ == '__main__':
    
    # desired history length
    hl = 64
    
    # initialize and fit range normalize transform object
    tf = RangeNormalize()
    tf.fit(
        pos_mins = torch.Tensor([-50, -50]), pos_maxs = torch.Tensor([150, 150])
        )
    
    # load in-maze data
    in_input = tf.transform(torch.load(f'Datasets/in_data_HL{hl}.pt'))
    
    # create bounds for train/validation/test split
    b0 = int(in_input.size(0)*.85)
    b1 = int(in_input.size(0)*.95)
        
    # load out-maze data
    out_tr_data, out_va_data, out_te_data = load_out_maze_datasets(
        'InMaze/out_trajectory.pt', history_length = hl, transform = tf,
        )
    
    # split into input and labels
    out_tr_input, out_tr_label = out_tr_data[:]
    out_va_input, out_va_label = out_va_data[:]
    out_te_input, out_te_label = out_te_data[:]
        
    # initialize in-maze classifier
    insideMaze = InMazeModelNN(hidden_layer_sizes = [24,24], feature_dim = 2, history_length = hl)
    #insideMaze.load_state_dict(torch.load(f'InMaze/insideMaze_AUC_2LayerMLP_state_dict_HL{hl}.pt'))
    
    # initialize trainer object
    trainer = TrainerAUROC(
        optimizer = torch.optim.SGD(insideMaze.parameters(), lr = 1e-3))
    # run training scheme
    trainer.train(
        model = insideMaze, train_input0 = out_tr_input, train_input1 = in_input[:b0],
        valid_input0 = out_va_input, valid_input1 = in_input[b0:b1],
        epochs = 100, plot_losses = True,
        )
    # save model state_dict
    torch.save(insideMaze.state_dict(), f'InMaze/insideMaze_AUC_2LayerMLP_state_dict_HL{hl}.pt')
    
    # create test input and test labels
    te_input = torch.cat([in_input[b1:], out_te_input], dim = 0)
    te_label = torch.cat([torch.ones((in_input[b1:].size(0),)), out_te_label], dim = 0)
    
    # generate test predictions
    te_pred_prob = insideMaze(te_input)
    te_pred = insideMaze.predict(te_input)
    
    # compute and print test AUC
    te_auc = roc_auc_score(te_label.numpy(), te_pred_prob.detach().numpy())
    print('\nTest AUC: ', te_auc)
    
    te_acc = (te_pred == te_label).float().mean().item()
    print('\nTest accuracy: ', te_acc)




