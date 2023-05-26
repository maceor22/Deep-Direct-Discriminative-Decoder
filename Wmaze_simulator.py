from maze_utils import Maze
from in_maze_model import InMazeModelNN
from data_generation import generate_trajectory_history
from state_process_models import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


# method for simulating trajectory using generative trajectory model
def trajectory_simulator(
        model, inside_maze_model, transform, init_sample, n_points, 
        Wmaze, threshold = 0.9, plot_trajectory = False, random_walk = False):
    # model: generative trajectory model
    # inside_maze_model: classifier model producing probability of trajectory being inside maze
    # transform: fitted transform object used to transform raw data
    # init_sample: initial sample pulled from real maze data
    # n_points: number of points to attempt simulating
    # Wmaze: untransformed Wmaze data; used for plotting
    # threshold: probability threshold required to continue simulating trajectory
    # plot_trajectory: boolean indicating whether to plot simulated trajectory
    # random_walk: boolean indicating whether a random walk model is used to simulate trajectory
    
    # set x_prev equal to initial sample
    x_prev = init_sample
    
    # initialize history of probabilities
    prob_hist = torch.ones((1,init_sample.size(1),1))
    
    # initialize list to hold simulated data points
    pos = []
    
    # simulation
    for i in range(n_points):
        # create input to trajectory model
        input_ = torch.cat([x_prev, prob_hist], dim = -1).flatten(1,-1)
        
        if random_walk:
            var = torch.var(x_prev, dim = 1).squeeze()
        
        # initialize sample rejection counter
        count = 0
        
        # logic for resampling if threshold condition is not met
        done = False
        while not done:
            # generate prediction for next time step
            if random_walk:
                xi = model.predict(input_, var = var).unsqueeze(0)
            else:
                xi = model.predict(input_).unsqueeze(0)
            
            # determine probability of new trajectory being inside maze
            inquiry = torch.cat([x_prev[:,1:,:], xi], dim = 1)
            prob = inside_maze_model(inquiry)
            
            # update counter
            count += 1
            print('n_point: %d | prob: %.4f | iter: %d' % (i+1, prob.item(), count))
            
            # threshold condition
            if prob >= threshold:
                # update probability history
                prob = prob.unsqueeze(0).unsqueeze(1).unsqueeze(2)
                prob_hist = torch.cat([prob_hist[:,1:,:], prob], dim = 1)
                # end while loop
                done = True
            
            # break from infinite loop
            if count >= 100:
                break
        
        # break from for loop if infinite loop is encountered
        if count >= 100:
            break
        
        # save new data point to list
        pos.append(xi)
        # update x_prev
        x_prev = inquiry   
        
    # concatenate simulated points into trajectory and untransform
    pos = torch.cat(pos, axis = 0)
    pos = transform.untransform(pos).squeeze(1).detach().numpy()
    
    # plotting
    if plot_trajectory:
        dt = 0.033
        time = np.arange(pos.shape[0]) * dt    
        
        fig, ax = plt.subplots(2,1, sharex = True)
        
        plt.xlabel('Time [s]')
        
        ax[0].plot(time, pos[:,0], 'k')
        ax[0].set_ylabel('X position')
        
        ax[1].plot(time, pos[:,1], 'k')
        ax[1].set_ylabel('Y position')
        
        fig.suptitle('Trajectory Simulated by Model')
        plt.show()
        
        
        plt.figure()
        plt.plot(Wmaze[:,0], Wmaze[:,1], '0.4')
        plt.plot(pos[:,0], pos[:,1], 'red')
        
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Maze Shape (X-Y) view')
        plt.show()
        
    return pos


# random walk model
class RandomWalk(nn.Module):
    
    def __init__(self, xvar = 0.2857/200, yvar = 0.6330/200):
        super(RandomWalk, self).__init__()
        
        covar = torch.zeros((2,2))
        covar[0,0] = xvar
        covar[1,1] = yvar
        self.covar = covar

    def forward(self, x, var = None):
        x = x[:,-3:-1]
        
        if var is None:
            return MultivariateNormal(x, self.covar).sample()
        else:
            covar = torch.zeros((2,2))
            covar[0,0] = var[0]
            covar[1,1] = var[1]
            return MultivariateNormal(x, covar).sample()
            
    
    def predict(self, x, var = None):
        return self.forward(x, var)



if __name__ == '__main__':
    
    # load maze data
    wm = Maze(
        name = 'Remy',
        session = (36,1),
        n_points = 1500000,
        Fs = 3000,
        rem_insig_chans = True,
        threshold = 0,
        )
    
    # initialize and fit range normalize transform object
    tf = RangeNormalize()
    tf.fit(
        pos_mins = torch.Tensor([-50, -50]), pos_maxs = torch.Tensor([150, 150])
        )
    
    # desired history length
    hl = 16
    
    # load position data with desired history length and transform
    dat = tf.transform(
        wm.generate_position_history(history_length = hl)[:,:,:2]
        )
    
    # boolean indicating whether to use generative trajectory model or random walk
    use_trajectory_model = True
    
    if use_trajectory_model:
        # initialize modelA and load state_dict
        P_x_given_input = PositionNN(hidden_layer_sizes = [32,32], input_dim = hl*3)
        P_x_given_input.load_state_dict(
            torch.load(f'Models/P_x_y_HL{hl}/2layerMLP_P_x_given_input_hid32.pt'))
        # initialize modelB and load state_dict
        P_y_given_x_input = PositionNN(hidden_layer_sizes = [32,32], input_dim = hl*3+2)
        P_y_given_x_input.load_state_dict(
            torch.load(f'Models/P_x_y_HL{hl}/2layerMLP_P_y_given_x_input_hid32.pt'))
        # load modelA and modelB into two-model wrapper object
        JointProb_x_y = TwoModelWrapper(
            modelA = P_x_given_input, modelB = P_y_given_x_input,
            transform = tf,
            )
        model = JointProb_x_y
    
    else:
        # initialize random walk
        # this implementation still needs work
        model = RandomWalk()

    # initialize and load in-maze classifier
    insideMaze = InMazeModelNN(hidden_layer_sizes = [24,24], feature_dim = 2, history_length = hl)
    insideMaze.load_state_dict(
        torch.load(f'InMaze/insideMaze_AUC_2LayerMLP_state_dict_HL{hl}.pt'))
    
    
    # get untransformed maze data for plotting purposes
    init_threshold = 0.85
    Wmaze = tf.untransform(dat[:,-1,:].unsqueeze(1)).squeeze()
    
    # boolean toggle indicating whether to produce plot of possible 
    #   initial samples given init_threshold
    plot_possible_inits = True
    
    if plot_possible_inits:
        # generate in-maze classifier probabilities        
        probs = insideMaze(dat)
        # get indexes of trajectories with in-maze classifier probability 
        #   greater than init_threshold
        idxs = torch.nonzero(torch.where(probs > init_threshold, 1, 0))
        # get points from maze data
        inq = Wmaze[idxs,:].squeeze()
        # plotting
        plt.figure()
        plt.xlabel('X-Axis')
        plt.ylabel('Y-Axis')
        plt.title('Possible Initial Starting Points for Simulation')
        plt.plot(Wmaze[:,0], Wmaze[:,1], '0.4')
        plt.plot(inq[:,0], inq[:,1], 'o', color = 'red', label = 'possible inits')
        plt.legend(fontsize = 8)
        plt.show()
        
    
    # find initial sample above threshold init_threshold
    print('finding appropriate initial sample.....')
    done = False
    while not done:
        # randomly sample data point from maze
        init_sample = dat[torch.randint(dat.size(0), (1,)),:,:]
        # generate in-maze classifier probability of initial sample being inside maze
        prob = insideMaze(init_sample)
        # threshold condition
        if prob > init_threshold:
            # break from while loop
            done = True
    print('  sample found.....\n')
    
    
    # run trajectory simulation
    traj = trajectory_simulator(
        model = model, 
        inside_maze_model = insideMaze, 
        transform = tf, 
        init_sample = init_sample, 
        n_points = 1000, 
        Wmaze = Wmaze, 
        threshold = 0.8, 
        plot_trajectory = True, 
        random_walk = not use_trajectory_model,
        )    















