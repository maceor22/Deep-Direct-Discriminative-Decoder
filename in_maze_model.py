import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import time
from copy import deepcopy
from maze_utils import RangeNormalize
from state_process_models import Data
from data_generation import generate_trajectory_history, data_to_index


# method for generating history of in-maze probabilities
def generate_inside_maze_prob_history(data, inside_maze_model):
    # data: trajectory data
    # inside_maze_model: in-maze classifier
    
    prob_hist = torch.zeros((data.size(0), data.size(1), 1))
    
    for h in range(data.size(1)):
        print(h)
        prob_hist[:,h,:] = inside_maze_model(data[:,h,:]).unsqueeze(1)
    
    return torch.cat([data, prob_hist], dim = 2)
    

# MLP classifier model producing probability that a given trajectory is inside the maze
class InMazeModelNN(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, feature_dim, history_length, 
            activation = 'relu', log_prob_model = False
            ):
        # hidden_layer_sizes: list or tuple containing sizes of hidden layers
        # feature_dim: number of features in input
        # history_length: length of input sequence dimension
        # log_prob_model: boolean indicating whether model is being trained
        #   with negative log-likelihood loss
        
        super(InMazeModelNN, self).__init__()
        
        self.log_prob_model = log_prob_model
                
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, feature_dim*history_length)
        
        if activation == 'relu':
            activ = nn.ReLU()
        elif activation == 'sigmoid':
            activ = nn.Sigmoid()
        elif activation == 'tanh':
            activ = nn.Tanh()
        elif activation == 'hardshrink':
            activ = nn.Hardshrink()
        
        # build hidden layers with specified activation function
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(activ)
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(layer_sizes[-1], 2 if log_prob_model else 1))
        
        self.lin = nn.Sequential(*layers)
        self.final = nn.LogSoftmax(dim = 1) if log_prob_model else nn.Sigmoid()

        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x)).squeeze()
    
    # method for outputting binary class (0 for out-maze, 1 for in-maze)
    def predict(self, x, return_prob = False):
        out = self.forward(x)
        if self.log_prob_model:
            if return_prob:
                return torch.exp(out[:,1]) if out.dim() == 2 else torch.exp(out[1])
            else:
                return out.argmax(dim = 1) if out.dim() == 2 else out.argmax()
        else:
            if return_prob:
                return self.forward(x)
            else:
                return torch.where(self.forward(x) > 0.5, 1, 0)


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
    

# class for training classifier model using BCE
class TrainerBCE(object):
    
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
        
        best_loss = 1e2
            
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
                clip_grad_value_(model.parameters(), clip_value = 1.0)
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
        
        best_acc = 0
            
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
                clip_grad_value_(model.parameters(), clip_value = 1.0)
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
                
                # compute validation accuracy 
                valid_input = torch.cat([va_input0, valid_input1], dim = 0)
                valid_pred = model.predict(valid_input)
                valid_acc = (valid_pred == torch.from_numpy(va_auc_target)).float().mean()
                
            # save model that performs best on validation data
            if valid_acc > best_acc:
                best_acc = valid_acc
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
    
    
# class for training classifier model using negative log-likelihood
class TrainerNLL(object):
    
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
        train_label = torch.cat([tr_label0, tr_label1], dim = 0).long()
        
        va_label0 = torch.zeros((valid_input1.size(0),))
        va_label1 = torch.ones((valid_input1.size(0),))
        valid_label = torch.cat([va_label0, va_label1], dim = 0).long()
        
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = 1e2
            
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
                
                log_prob = model(input_batch)
                
                loss = nn.NLLLoss()(log_prob, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(model.parameters(), clip_value = 1.0)
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
                    log_prob = model(input_batch)
                    # compute and aggregate validation losses
                    valid_loss += nn.NLLLoss()(log_prob, label_batch).item()
            
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
            plt.ylabel('Negative Log Likelihood')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses
    


class GridClassifier(nn.Module):
    
    def __init__(self, grid, xmin, ymin, resolution, transform):
        super(GridClassifier, self).__init__()
        
        self.grid = grid
        self.xmin = xmin
        self.ymin = ymin
        self.resolution = resolution
        self.tf = transform
    
    def forward(self, x, untransform = True):
        dat = x.unsqueeze(0) if x.dim() == 1 else x
        dat = self.tf.untransform(dat)
        dat = dat.unsqueeze(0) if dat.dim() == 1 else dat
        ix = data_to_index(
            dat, self.xmin, self.ymin, self.resolution, unique = False)
        
        if x.dim() == 1:
            xix, yix = ix[0,0], ix[0,1]
            out = self.grid[xix,yix]
        else:
            out = []
            for i in range(ix.size(0)):
                xix, yix = ix[i,0], ix[i,1]
                out.append(self.grid[xix,yix].unsqueeze(0))
            out = torch.cat(out, dim = 0)
        return out
    
    def predict(self, x, untransform = True):
        return self.forward(x, untransform)



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
    
    # load data, generate trajectory history with desired history length, and shuffle
    out_data = torch.load(path)
    input_data = generate_trajectory_history(out_data, history_length)
    input_data = input_data[torch.randperm(input_data.size(0))]
    
    if transform is not None:
        # apply transform
        input_data = transform.transform(input_data)
    
    input_data = input_data.squeeze()
    
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



def generate_heat_map(
        inside_maze_model, transform, 
        xmin, xmax, ymin, ymax, 
        resolution = 0.01, plot_heat_map = True,
        ):
    
    xpoints = 0.5*resolution*(2*torch.arange(1, int((xmax-xmin)/resolution)+1)-1) + xmin
    ypoints = 0.5*resolution*(2*torch.arange(1, int((ymax-ymin)/resolution)+1)-1) + ymin
    
    grid = torch.zeros((xpoints.size(0), ypoints.size(0)))

    for i in range(xpoints.size(0)):
        for j in range(ypoints.size(0)):
            point = torch.cat([xpoints[i].unsqueeze(0), ypoints[j].unsqueeze(0)], 
                              dim = 0).unsqueeze(0)
            prob = inside_maze_model.predict(
                transform.transform(point), return_prob = True)
            print(i, j, point.size(), prob.size())
            grid[i,j] = prob
    
    if plot_heat_map:
        fig, ax = plt.subplots()
        fig.suptitle('In-Maze Probability Heat Map')
        img = ax.pcolormesh(grid.detach().numpy(), cmap = 'plasma')
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        xticks = np.arange(0, int((xmax-xmin)/resolution)+1, int((xmax-xmin)/(resolution*10)))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks*resolution+xmin)
        yticks = np.arange(0, int((ymax-ymin)/resolution)+1, int((ymax-ymin)/(resolution*10)))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks*resolution+ymin)
        plt.colorbar(img, ax = ax)
    
    
    return


if __name__ == '__main__':
    
    tf = RangeNormalize()
    tf.fit(
        pos_mins = torch.Tensor([-50, -50]), pos_maxs = torch.Tensor([150, 150])
        )
    
    insideMaze = InMazeModelNN(
        hidden_layer_sizes = [32,32], feature_dim = 2, 
        history_length = 1, log_prob_model = True,
        )
    insideMaze.load_state_dict(torch.load('InMaze/tanh_NLL_2LayerMLP_state_dict_HL1.pt'))
    
    generate_heat_map(
        insideMaze, tf, xmin = -50, xmax = 150, ymin = -50, ymax = 150,
        resolution = 2,
        )
    
# =============================================================================
#     # desired history length
#     hl = 1
#     training_scheme = 'NLL'
#     activation = 'hardshrink'
#     
#     # initialize and fit range normalize transform object
#     tf = RangeNormalize()
#     tf.fit(
#         pos_mins = torch.Tensor([-50, -50]), pos_maxs = torch.Tensor([150, 150])
#         )
#     
#     # load in-maze data and shuffle
#     in_input = tf.transform(torch.load(f'Datasets/in_data_HL{hl}.pt')).squeeze()
#     in_ix = torch.randperm(in_input.size(0))
#     in_input = in_input[in_ix]
#     
#     # create bounds for train/validation/test split
#     if training_scheme == 'AUC':
#         b0 = int(in_input.size(0)*.15)
#         b1 = int(in_input.size(0)*.20)
#     elif training_scheme == 'BCE' or training_scheme == 'NLL':
#         b0 = int(in_input.size(0)*.80)
#         b1 = int(in_input.size(0)*.90)
#     
#     # load out-maze data
#     out_tr_data, out_va_data, out_te_data = load_out_maze_datasets(
#         'InMaze/out_classifier.pt', history_length = hl, transform = tf,
#         )
#     
#     # split into input and labels
#     out_tr_input, out_tr_label = out_tr_data[:]
#     out_va_input, out_va_label = out_va_data[:]
#     out_te_input, out_te_label = out_te_data[:]
#     
#     
#     # initialize in-maze classifier
#     insideMaze = InMazeModelNN(
#         hidden_layer_sizes = [32,32], feature_dim = 2, history_length = hl, 
#         activation = activation, log_prob_model = (training_scheme == 'NLL'))
#     #insideMaze.load_state_dict(torch.load(f'InMaze/insideMaze_{training_scheme}_2LayerMLP_state_dict_HL{hl}.pt'))
#     
#     # initialize trainer object
#     if training_scheme == 'AUC':
#         trainer = TrainerAUROC(
#             optimizer = torch.optim.SGD(insideMaze.parameters(), lr = 1e-3))
#     elif training_scheme == 'BCE':
#         trainer = TrainerBCE(
#             optimizer = torch.optim.SGD(insideMaze.parameters(), lr = 1e-3))
#     elif training_scheme == 'NLL':
#         trainer = TrainerNLL(
#             optimizer = torch.optim.SGD(insideMaze.parameters(), lr = 1e-3))
#     # run training scheme
#     trainer.train(
#         model = insideMaze, train_input0 = out_tr_input, train_input1 = in_input[:b0],
#         valid_input0 = out_va_input, valid_input1 = in_input[b0:b1],
#         epochs = 500, batch_size = 256, plot_losses = True,
#         )
#     # save model state_dict
#     torch.save(insideMaze.state_dict(), f'InMaze/{activation}_{training_scheme}_2LayerMLP_state_dict_HL{hl}.pt')
#     
#     # create test input and test labels
#     in_te_input = in_input[b1:]
#     in_te_input = in_te_input[torch.randperm(in_te_input.size(0))[:out_te_input.size(0)]]
#     te_input = torch.cat([in_te_input, out_te_input], dim = 0)
#     te_label = torch.cat([torch.ones((in_te_input.size(0),)), out_te_label], dim = 0)
#     
#     # generate test predictions
#     te_pred_prob = insideMaze.predict(te_input, return_prob = True)
#     te_pred = insideMaze.predict(te_input, return_prob = False)
#     
#     # compute and print test AUC and accuracy
#     te_auc = roc_auc_score(te_label.numpy(), te_pred_prob.detach().numpy())
#     print('\nTest AUC: ', te_auc)
#     
#     sep = int(te_pred.size(0)/2)
#     te_acc = (te_pred == te_label).float().mean().item()
#     in_te_acc = (te_pred[:sep] == te_label[:sep]).float().mean().item()
#     out_te_acc = (te_pred[sep:] == te_label[sep:]).float().mean().item()
#     print('\n   Total test accuracy: ', te_acc)
#     print(' In-maze test accuracy: ', in_te_acc)
#     print('Out-maze test accuracy: ', out_te_acc)
# 
# # =============================================================================
# #     generate_heat_map(
# #         inside_maze_model = insideMaze, 
# #         xmin = -50, xmax = 150, ymin = -50, ymax = 150,
# #         resolution = 2,)
# # =============================================================================
# =============================================================================

