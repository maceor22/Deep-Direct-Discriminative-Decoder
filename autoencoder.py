import torch
from torch import nn as nn
from torch import autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
from torch.distributions import MultivariateNormal, Normal
import time
from copy import deepcopy
from maze_utils import RangeNormalize
from state_process_models import Data
import numpy as np
import matplotlib.pyplot as plt



class AutoEncoder(nn.Module):
    
    def __init__(
            self, feature_dim, encoding_layers, 
            bottleneck_layer, decoding_layers,
            ):
        
        super(AutoEncoder, self).__init__()
        
        encoding_layers.insert(0, feature_dim)
        encoding = []
        for i in range(1, len(encoding_layers)-1):
            encoding.append(nn.Linear(encoding_layers[i-1], encoding_layers[i]))
            encoding.append(nn.Hardshrink(lambd = 0))
            encoding.append(nn.Dropout(0.5))
        encoding.append(nn.Linear(encoding_layers[-2], encoding_layers[-1]))
        encoding.append(nn.Hardshrink(lambd = 0))
        encoding.append(nn.Dropout(0.5))
        encoding.append(nn.Linear(encoding_layers[-1], bottleneck_layer))
        
        decoding_layers.insert(0, bottleneck_layer)
        decoding = []
        decoding.append(nn.Hardshrink(lambd = 0))
        for i in range(1, len(decoding_layers)-1):
            decoding.append(nn.Linear(decoding_layers[i-1], decoding_layers[i]))
            decoding.append(nn.Hardshrink(lambd = 0))
            decoding.append(nn.Dropout(0.5))
        decoding.append(nn.Linear(decoding_layers[-2], decoding_layers[-1]))
        
        final_layer = []
        final_layer.append(nn.Hardshrink(lambd = 0))
        final_layer.append(nn.Dropout(0.5))
        final_layer.append(nn.Linear(
            decoding_layers[-1], feature_dim))
        
        self.encode = nn.Sequential(*encoding)
        self.decode = nn.Sequential(*decoding)
        self.final = nn.Sequential(*final_layer)
        self.output = 'final'
        
    def forward(self, x):
        x = self.encode(x)
        if self.output == 'code':
            return x
        x = self.decode(x)
        if self.output == 'last':
            return x
        x = self.final(x)
        if self.output == 'final':
            return x
        
    def set_output(self, output):
        self.output = output



# class for training model using Gaussian negative log-likelihood
class TrainerAutoEncoder(object):
    
    def __init__(self, optimizer):
        # optimizer: optimizer object used during training
        self.optimizer = optimizer
    
    # method for training
    def train(self, model, transform, train_input, valid_input, 
              epochs = 100, batch_size = 64, plot_losses = False):
        # model: initialized model to be trained
        # train_data: Data object containing training data
        # valid_data: Data object containing validation data
        
        # send model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # create dataloader objects for training data and validation data
        train_data = Data(transform.transform(train_input), train_input)
        valid_data = Data(transform.transform(valid_input), valid_input) 
        train_dataloader = DataLoader(
            train_data, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(
            valid_data, batch_size = batch_size, shuffle = True)
        
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
                pred = transform.untransform(model(input_batch))
                # compute loss
                loss = nn.MSELoss()(pred, label_batch)
                
                # zero gradients
                self.optimizer.zero_grad()
                # backpropagate loss
                loss.backward()
                # prevent exploding gradients
                clip_grad_value_(model.parameters(), clip_value = 10.0)
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
                    pred = transform.untransform(model(input_batch))
                    # compute and aggregate validation loss 
                    valid_loss += nn.MSELoss()(pred, label_batch).item()
            
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
            plt.ylabel('Mean Square Error')
            plt.title('Loss Curves')
            plt.legend()
            plt.show()
            
        return best_epoch, train_losses, valid_losses



if __name__ == '__main__':
    
# =============================================================================
#     included_data = 'maze'
#     
#     if included_data == 'all':
#         
#         in_data = torch.load('Datasets/in_data_HL1.pt').squeeze()
#         in_data = in_data[torch.randperm(in_data.size(0))]
#         out_data = torch.load('InMaze/out_classifier.pt')
#         out_data = out_data[torch.randperm(out_data.size(0))]
#         
#         b0 = 200000
#         b1 = 300000
#         
#         train_input = torch.cat([in_data[:b0], out_data[:b0]], dim = 0)
#         valid_input = torch.cat([in_data[b0:b1], out_data[b0:b1]], dim = 0)
#         test_input = torch.cat([in_data[b1:], out_data[b1:]], dim = 0)
#     
#     elif included_data == 'maze':
#         
#         in_data = torch.load('Datasets/in_data_HL1.pt').squeeze()
#         in_data = in_data[torch.randperm(in_data.size(0))]
#         
#         b0 = 400000
#         b1 = 450000
#         
#         train_input = in_data[:b0]
#         valid_input = in_data[b0:b1]
#         test_input = in_data[b1:]
#         
# 
#     autoencoder = AutoEncoder(
#         feature_dim = 2, encoding_layers = [32,16,16,8], 
#         bottleneck_layer = 1, decoding_layers = [8,16,16,32],
#         )
# 
#     trainer = TrainerAutoEncoder(
#         optimizer = torch.optim.SGD(autoencoder.parameters(), lr = 1e-3))
#     
#     tf = RangeNormalize()
#     tf.fit(
#         pos_mins = torch.tensor([-50, -50]), pos_maxs = torch.tensor([150, 150])
#         )
#     
#     trainer.train(
#         model = autoencoder, transform = tf,
#         train_input = train_input, valid_input = valid_input,
#         epochs = 500, batch_size = 256, plot_losses = True,
#         )
# 
#     torch.save(autoencoder.state_dict(), f'InMaze/{included_data}_hs_autoencoder_state_dict.pt')
# 
# 
#     test_pred = tf.untransform(autoencoder(tf.transform(test_input)))
# # =============================================================================
# #     test_pred = autoencoder(tf.transform(test_input.unsqueeze(1)).squeeze())
# #     test_pred = Normal(test_pred[:,:2], torch.abs(test_pred[:,2:])).sample()
# #     test_pred = tf.untransform(test_pred.unsqueeze(1)).squeeze()
# # =============================================================================
#     
#     print(test_input)
#     print(test_pred)    
#     MSE = nn.MSELoss()(test_pred, test_input)
#     
#     print('Autoencoder test MSE: %.4f' % MSE)
# =============================================================================

    
    autoencoder = AutoEncoder(
        feature_dim = 2, encoding_layers = [32,16,16,8], 
        bottleneck_layer = 1, decoding_layers = [8,16,16,32],
        )
    autoencoder.load_state_dict(torch.load('InMaze/maze_hs_autoencoder_state_dict.pt'))
    
    in_data = torch.load('Datasets/in_data_HL1.pt').squeeze()
    in_data = in_data[torch.randperm(in_data.size(0))][:10000]
    out_data = torch.load('InMaze/out_classifier.pt')
    out_data = out_data[torch.randperm(out_data.size(0))][:10000]
    
    
    MSE = nn.MSELoss()(autoencoder(in_data), in_data).item()
    print('MSE: ', MSE)
    
    
    autoencoder.set_output('code')
    
    in_code = autoencoder(in_data).detach()
    out_code = autoencoder(out_data).detach()
    print(in_code.size(), out_code.size())
    
    plt.figure()
    plt.plot(out_code, out_code, 'o', color = 'orange', markersize = 5, label = 'out-maze')
    plt.plot(in_code, in_code, 'o', color = 'blue', markersize = 0.5, label = 'in-maze')
    plt.legend()
    plt.xlabel('Code Dimension')
    plt.ylabel('Code Dimension')
    plt.title('Linear Separability of Encoded Data | Maze Data')





