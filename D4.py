from maze_utils import *
from models import *
from misc_plotting import *
from in_maze_model import GridClassifier
from state_process import StateProcess1, StateProcess
from observation_process import ObservationProcess
from data_generation import generate_dataset
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
import time as time


class DeepDirectDiscriminativeDecoder(nn.Module):

    def __init__(
            self, observation_process, state_process, history_length, epsilon = 1e-9
    ):
        super(DeepDirectDiscriminativeDecoder, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.observation_process = observation_process.to(device)
        self.state_process = state_process.to(device)
        self.history_length = history_length
        self.epsilon = epsilon
        self.device = device


    def importance_resample(self, yt, unweighted_particles):
        # compute probability of particles given observation
        observation_prob = self.observation_process.prob(
            yt.expand(self.n_particles, -1, -1), unweighted_particles)
        print((observation_prob < 0).sum())
        # combine observation prob with in-maze prob of unweighted particles
        temp = self.state_process.clf(unweighted_particles.to('cpu')).to(self.device)
        self.inmaze_particles.append(temp.sum(0).item())
        print(temp.sum())
        if temp.sum(0) == 0:
            prob = observation_prob + self.epsilon
        else:
            prob = observation_prob*temp + self.epsilon
        # compute weights based on combined probability
        W = (prob / prob.sum(dim = 0))
        print(W.isnan().any(), W.isinf().any())
        # resample particles based on weights
        ix = torch.multinomial(W, num_samples = self.n_particles, replacement = True)
        return ix

    # only supports single timestep as input
    def forward(self, yt, particle_trace):
        # obtain target distribution of next particles and sample
        unweighted_particles = self.state_process.predict(
            particle_trace[:,-(self.history_length+1):,:])
        
        weighted_ix = self.importance_resample(yt, unweighted_particles)
        weighted_particles = unweighted_particles[weighted_ix]

        #print(yt.device, unweighted_particles.device, weighted_ix.device, weighted_particles.device)

        self.particles.append(unweighted_particles.to('cpu').detach())
        self.particles.append(weighted_particles.to('cpu').detach())
        self.traces.append(particle_trace.to('cpu').detach())

        particle_trace = torch.cat([
            particle_trace[weighted_ix], weighted_particles.unsqueeze(1)], dim = 1)
        
        self.traces.append(particle_trace.to('cpu').detach())

        return particle_trace
    

    def filter(self, observations, n_particles = 1, smoother = False):
        observations = observations.to(self.device)

        self.n_particles = n_particles
        
        self.inmaze_particles = []
        self.particles = []
        self.traces = []

        print('\nfiltering.....')
        y0 = observations[0]
        
        unweighted_particles = self.observation_process.predict(
        y0.expand(n_particles, -1, -1), return_sample = True)

        weighted_ix = self.importance_resample(y0, unweighted_particles)
        weighted_particles = unweighted_particles[weighted_ix]

        filter_trace = weighted_particles.expand(
            self.history_length+1, -1, -1).transpose(0,1)
        
        self.particles.append(unweighted_particles.to('cpu').detach())
        self.particles.append(weighted_particles.to('cpu').detach())
        self.traces.append(
            unweighted_particles.expand(self.history_length+1, -1, -1
                                        ).transpose(0,1).to('cpu').detach())
        self.traces.append(filter_trace.to('cpu').detach())

        # filter recursion
        for t in range(1, observations.size(0)):
            print(t)
            yt = observations[t]
            # obtain weighted particles and marginal likelihood
            filter_trace = self.forward(yt, filter_trace)
        
        filter_trace = filter_trace[:,self.history_length:,:]

        if smoother:
            smoother_trace = self.smoother(observations, filter_trace)
            return filter_trace, smoother_trace
        
        else:
            return filter_trace
    

    def smoother(self, observations, filter_trace):
        smoother_trace = filter_trace

        print('\nsmoothing.....')
        for t in range(observations.size(0)-1,-1,-1):
            yt = observations[t]

            weighted_ix = self.importance_resample(yt, smoother_trace[:,t,:])

            smoother_trace = smoother_trace[weighted_ix]

        return smoother_trace



if __name__ == '__main__':

    shl = 20
    bs = 1
    phl = 0

    grid_res = 4
    bal_res = 0.1
    
    data = generate_dataset(
        rat_name = 'Bon', 
        input_history_length = shl, spike_bin_size = bs, label_history_length = 0, 
        include_velocity = False, dirvel = False, dmin = 0.5, dmax = 20,
        grid_resolution = grid_res, balance_resolution = bal_res, 
        threshold = 100, presence = False, p_range = 2,
        )
    
    maze_grid = data['maze_grid']
    
    raw_observations, latent = data['test_spikes'], data['test_labels'].float()
    
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']


    """ test_ix = data_to_index(latent, xmin, ymin, grid_res)
    for i in range(test_ix.size(0)):
        xix, yix = test_ix[i,0], test_ix[i,1]
        maze_grid[xix,yix] = 0.5
    
    plt.imshow(maze_grid)
    plt.show() """
    
    
    spikeNorm = RangeNormalize(dim = raw_observations.size(2), norm_mode = 'auto')
    spikeNorm.fit(
        range_min = [0 for _ in range(raw_observations.size(2))], 
        range_max = [10 for _ in range(raw_observations.size(2))],
        )
    
    observations = spikeNorm.transform(raw_observations)
    

    obs_root = 'ObservationModels/trained/MultivariateNormalMixtureMLP_2023-10-2_1-7-53'

    P_X__Y_H = MultivariateNormalMixtureMLP(
        hidden_layer_sizes = [24, 24], 
        num_mixtures = 5, 
        input_dim = observations.size(1) * observations.size(2), 
        latent_dim = 2,
        )
    P_X__Y_H.load_state_dict(torch.load(obs_root+'/P_X__Y_H/state_dict.pt'))

    P_X__H = MultivariateNormalMixtureMLP(
        hidden_layer_sizes = [24, 24], 
        num_mixtures = 5, 
        input_dim = (observations.size(1) - 1) * observations.size(2), 
        latent_dim = 2,
        )
    P_X__H.load_state_dict(torch.load(obs_root+'/P_X__H/state_dict.pt'))

    observation_process = ObservationProcess(P_X__Y_H = P_X__Y_H, P_X__H = P_X__H)


    posNorm = RangeNormalize(dim = 2, norm_mode = [0,0,])
    posNorm.fit(
        range_min = (xmin, ymin,), range_max = (xmax, ymax,),
        )
    
    insideMaze = GridClassifier(
        grid = maze_grid, xmin = xmin, ymin = ymin, 
        resolution = grid_res, transform = posNorm,
    )

    transition_model = GaussianMLP(
        hidden_layer_sizes = [16,16], 
        input_dim = (phl+1)*3, 
        latent_dim = latent.size(1),
    )
    transition_model.load_state_dict(torch.load(
        #'StateModels/trained/GaussianMLP_2023-11-1_4-15-12/state_dict.pt'
        #'StateModels/trained/GaussianMLP_2023-9-26_14-39-26/state_dict.pt'
        'StateModels/trained/GaussianMLP_2023-11-3_23-15-58/state_dict.pt'
    ))

    #transition_model = RandomWalk(xvar = 1e-5, yvar = 1e-5)

    state_process = StateProcess1(
        quality_classifier = insideMaze,
        #maze_classifier = None, 
        transition_model = transition_model, 
    )


    D4 =  DeepDirectDiscriminativeDecoder(
        observation_process = P_X__Y_H,
        state_process = state_process,
        history_length = phl,
    )

    filter_trace = D4.filter(
        observations = observations[550:1050], n_particles = 500, smoother = False)
    filter_trace = posNorm.untransform(filter_trace.to('cpu').detach())
    #smoother_trace = posNorm.untransform(smoother_trace)

    mse = nn.MSELoss()(filter_trace.mean(0), latent[550:1050]).item()
    print('\ntest mse:', mse)

    fig = plot_model_predictions(
        pred_mean = filter_trace.mean(dim = 0), pred_vars = filter_trace.var(dim = 0),
        label_data = latent[550:1050], title = 'Filter Performance | MSE: %.3f' % mse,
        )
    
    plt.figure()
    plt.plot(np.arange(len(D4.inmaze_particles))*0.033, D4.inmaze_particles,'k')
    plt.xlabel('Time')
    plt.ylabel('# of In-Maze Particles')
    plt.show()


    filter_vid = plot_filter_performance(D4, latent[550:1050], posNorm)
    plt.show()

    print('\nsaving filter video.....')
    filter_vid.save('D4 Eval/filter_test3.mp4', writer = 'ffmpeg', dpi = 200)

    """ LL_fig, LL_vid, _ = probabilistic_model_likelihood(
        model = P_X__Y_H, model_name = 'P(X|Y,H)', 
        input_data = observations[550:1050], label_data = latent[550:1050], 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 50, plotting = True, video = True,
        )

    LL_vid.save('D4 Eval/likelihood_test3.mp4', writer = 'ffmpeg', dpi = 200) """


    """ plot_model_predictions(
        pred_mean = smoother_trace.mean(dim = 0), pred_vars = smoother_trace.var(dim = 0),
        label_data = latent, title = 'Smoother Performance',
        )

    plt.show() """











