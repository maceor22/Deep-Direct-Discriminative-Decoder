from maze_utils import *
from models import *
from misc_plotting import *
from in_maze_model import GridClassifier
from data_generation import generate_dataset
import torch
from torch import nn as nn
import time as time
#from memory_profiler import profile 



class DeepDirectDiscriminativeDecoder(nn.Module):

    def __init__(
            self, prediction_model,
            latent_dim, penalty_model = None, 
            state_covariance_type = 'diag',
            transition_variance = None,
            proposal_variance_init = None,
            epsilon = 1e-20,
    ):
        super(DeepDirectDiscriminativeDecoder, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.prediction = prediction_model.to(device)
        for param in self.prediction.parameters():
            param.requires_grad = False

        """ self.transition = transition_model.to(device)
        for param in self.transition.parameters():
            param.requires_grad = False """
        self.penalty = penalty_model
        for param in self.penalty.parameters():
            param.requires_grad = False
        
        if state_covariance_type == 'diag':
            if proposal_variance_init is not None:
                Q_tril_components = (proposal_variance_init**0.5).log()
            else:
                Q_tril_components = torch.zeros((latent_dim,)).uniform_(-4,-2)
        elif state_covariance_type == 'full':
            Q_tril_components = torch.zeros((int((latent_dim * (latent_dim+1))/2),))
            diag_ix = torch.arange(1, latent_dim+1).cumsum(0).long() - 1

            if proposal_variance_init is not None:
                Q_tril_components[diag_ix] = (proposal_variance_init**0.5).log()
            else:
                Q_tril_components[diag_ix] = torch.rand_like(diag_ix) - 4

        self.Q_tril_components = nn.Parameter(
            Q_tril_components, requires_grad = True).to(device)

        self.transition_var = transition_variance.to(device)

        self.latent_dim = latent_dim
        self.state_covariance_type = state_covariance_type
        self.epsilon = epsilon
        self.device = device
    

    def _proposal(self, x_prev):
        Q_cholesky_tril = torch.zeros(
            (self.latent_dim, self.latent_dim),
            device = self.Q_tril_components.device,
            #requires_grad = True,
            )

        if self.state_covariance_type == 'diag':
            for i in range(self.latent_dim):
                Q_cholesky_tril[i,i] = self.Q_tril_components[i].exp()

        elif self.state_covariance_type == 'full':
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        Q_cholesky_tril[i,j] = self.Q_tril_components[idx].exp()
                    else:
                        Q_cholesky_tril[i,j] = self.Q_tril_components[idx]
                    idx += 1
        
        batch_shape = x_prev.shape[:-1]
        Q = MultivariateNormal(
            loc = x_prev,
            scale_tril = Q_cholesky_tril.expand(*[_ for _ in batch_shape],-1,-1),
        )
        
        return Q
    

    def _denominator(self, x_prev, x_curr):        
        x_prev = x_prev.expand(x_curr.size(0), x_prev.size(0), -1)
        x_curr = x_curr.expand(x_prev.size(0), x_curr.size(0), -1).transpose(0,1)
        
        log_prob = torch.logsumexp(
            self._proposal(x_prev).log_prob(x_curr), 
            dim = -1,
            )
        return log_prob
    

    def _state(self, x_prev, x_curr):
        P = Normal(x_prev, scale = self.transition_var**0.5)
        state = P.log_prob(x_curr).sum(dim = -1)
        state = state - self._proposal(x_prev).log_prob(x_curr)
        return state


    def importance_resample(self, yt, unweighted_particles, particle_trace):
        # compute probability of particles given observation
        numer = self.prediction.log_prob(
            yt.expand(self.n_particles,-1,-1), unweighted_particles)
        
        denom = self._denominator(
            particle_trace[:,-1,:], unweighted_particles)
        
        state = self._state(
            particle_trace[:,-1,:], unweighted_particles)

        # combine observation prob with in-maze prob of unweighted particles
        quality = self.penalty(unweighted_particles.to('cpu')).to(self.device)
        #print('\n In-maze count: ', quality.sum())

        prob = (state + numer - denom).exp() * quality + self.epsilon
        #prob = (state + numer).exp() * quality + self.epsilon
        prob.nan_to_num(nan = self.epsilon)

        # compute weights based on combined probability
        W = (prob / prob.sum(dim = 0))
        # resample particles based on weights
        ix = torch.multinomial(W, num_samples = self.n_particles, replacement = True)
        return ix


    # only supports single timestep as input
    def forward(self, yt, particle_trace):
        # obtain target distribution of next particles and sample
        unweighted_particles = self._proposal(particle_trace[:,-1,:]).sample()
        
        weighted_ix = self.importance_resample(yt, unweighted_particles, particle_trace)
        weighted_particles = unweighted_particles[weighted_ix]

        self.u_particles.append(unweighted_particles.to('cpu').detach())
        self.w_particles.append(weighted_particles.to('cpu').detach())

        particle_trace = torch.cat([
            particle_trace[weighted_ix], weighted_particles.unsqueeze(1)], dim = 1)
        
        return particle_trace
    

    def filter(self, observations, n_particles = 1, smoother = False):
        observations = observations.to(self.device)

        self.n_particles = n_particles
        
        self.u_particles = []
        self.w_particles = []
        self.traces = []

        #print('\nfiltering.....')
        y0 = observations[0]
        
        particles_init = torch.distributions.Uniform(0,1).sample(
            (n_particles, self.latent_dim)).to(self.device)
        unweighted_particles = self._proposal(particles_init).sample()

        weighted_ix = self.importance_resample(
            y0, unweighted_particles, particles_init.unsqueeze(1))
        weighted_particles = unweighted_particles[weighted_ix]

        particle_trace = weighted_particles.unsqueeze(1)
        
        self.u_particles.append(unweighted_particles.to('cpu').detach())
        self.w_particles.append(weighted_particles.to('cpu').detach())

        # filter recursion
        for t in range(1, observations.size(0)):
            #print(t)
            yt = observations[t]
            # obtain weighted particles
            particle_trace = self.forward(yt, particle_trace)
        
        if smoother:
            return torch.stack(self.w_particles, dim = 1), particle_trace
        
        else:
            return torch.stack(self.w_particles, dim = 1)
        
    #@profile
    def log_prob(
            self, observations, latent, 
            n_particles = 100, component = 'denominator',
            ):
        
        if component in ['discriminative', 'ratio', 'full']:
            numer = self.prediction.log_prob(observations, latent)[1:]
        
        if component in ['denominator', 'ratio', 'full']:
            filter_trace = self.filter(observations, n_particles).to(self.device).detach()

            denom = torch.logsumexp(
                self._proposal(filter_trace.transpose(0,1)[:-1]).log_prob(latent[1:].expand(n_particles,-1,-1).transpose(0,1)),
                dim = -1,
                )
            denom = denom - torch.tensor(
                n_particles, dtype = denom.dtype, device = denom.device, requires_grad = denom.requires_grad,
                ).log()
        
        if component in ['transition', 'full']:
            state = self._state(latent[:-1], latent[1:])

        if component == 'numerator':
            return numer
        elif component == 'denominator':
            return denom
        elif component == 'ratio':
            return numer - denom
        elif component == 'state':
            return state
        elif component == 'full':
            return state + numer - denom



if __name__ == '__main__':

    shl = 8
    bs = 1
    phl = 0

    grid_res = 2
    bal_res = 0.1
    
    data = generate_dataset(
        rat_name = 'Bon', pos_history_length = phl, 
        spike_history_length = shl, spike_bin_size = bs, label_history_length = 0, 
        include_velocity = False, dirvel = False, dmin = 0.0, dmax = 100,
        grid_resolution = grid_res, balance_resolution = bal_res, 
        threshold = 100, presence = False, p_range = 2, 
        down_sample = True, up_sample = False,
        )
    
    maze_grid = data['maze_grid']
    
    raw_observations = data['raw_test_spikes'][:]
    latent = data['raw_test_labels'][:].float()
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']


    observations = torch.log(raw_observations + 1)


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
        input_dim = (phl+1)*2, 
        latent_dim = latent.size(1),
        covariance_type = 'diag',
    )

    transition_model.load_state_dict(torch.load(
        'StateModels/trained/GaussianMLP_2024-1-30_21-38-6/transition_state_dict.pt'
    ))
    

    obs_root = 'ObservationModels/trained/MVN-Mix-MLP_2024-2-1_1-48-9'

    P_x__y_h = GaussianMixtureMLP(
        hidden_layer_sizes = [32, 32], 
        num_mixtures = 5, 
        input_dim = observations.size(1) * observations.size(2), 
        latent_dim = 2,
        covariance_type = 'full',
        )
    P_x__y_h.load_state_dict(torch.load(obs_root+'/P_X__Y_H/state_dict.pt'))


    temp = posNorm.transform(latent)

    D4 = DeepDirectDiscriminativeDecoder(
        prediction_model = P_x__y_h, 
        penalty_model = insideMaze,
        latent_dim = latent.size(1),
        transition_variance = 2*(temp[1:] - temp[:-1]).var(0),
        state_covariance_type = 'diag',
        proposal_variance_init = 5*(temp[1:] - temp[:-1]).var(0),
    )

    """ filter_rmse = []
    for _ in range(10):
        print('filtering')
        filter_trace, smoother_trace = D4.filter(
            observations = observations, n_particles = 1000, smoother = True)
        filter_trace = posNorm.untransform(filter_trace.to('cpu').detach())
        filter_rmse.append(nn.MSELoss()(
            filter_trace, latent.expand(D4.n_particles,-1,-1)).item()**0.5)
        print(filter_rmse[-1])
    
    filter_rmse = torch.tensor(filter_rmse)
    print('Confidence Interval: ', filter_rmse.mean(0), '+/-', 2*filter_rmse.std(0)) """
        

    """ print('\nfiltering.....')
    filter_trace, smoother_trace = D4.filter(
        observations = observations, n_particles = 1000, smoother = True)
    print('\nfilter completed.....')
    
    filter_trace = posNorm.untransform(filter_trace.to('cpu').detach())
    smoother_trace = posNorm.untransform(smoother_trace.to('cpu').detach()) """
    
    """ data_mean, data_std = latent.mean(dim = 0), latent.std(dim = 0)
    
    filter_trace = (filter_trace - data_mean) / data_std

    latent = (latent - data_mean) / data_std """

    """ raster = raster_plot(observations[:,-1,:].T, latent)
    #maze = plot_maze(data['raw_train_labels'])
    plt.show() """

    """ filter_mse = nn.MSELoss()(
        filter_trace, latent.expand(D4.n_particles,-1,-1)).item()
    smoother_mse = nn.MSELoss()(
        smoother_trace, latent.expand(D4.n_particles,-1,-1)).item()
    print('\nfilter test mse:', filter_mse)
    print('\nsmoother test mse:', smoother_mse) """

    """ filter_fig = plot_model_predictions(
        pred_mean = filter_trace.mean(dim = 0), pred_vars = filter_trace.var(dim = 0),
        label_data = latent, title = 'Filter Performance | RMSE: %.3f' % filter_mse**0.5,
        )
    smoother_fig = plot_model_predictions(
        pred_mean = smoother_trace.mean(dim = 0), pred_vars = smoother_trace.var(dim = 0),
        label_data = latent, title = 'Smoother Performance | RMSE: %.3f' % smoother_mse**0.5,
        )
    
    #plt.show()

    filter_vid = plot_filter_performance(D4, latent, posNorm, filter_mse**0.5)
    
    now = datetime.datetime.now()
    root = f'D4 Eval/run_'
    root += f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
    
    os.mkdir(root)

    plt.show()

    raster.savefig(root+'/raster_plot.jpeg')
    maze.savefig(root+'/maze_shape.jpeg')
    filter_fig.savefig(root+'/filter_results.jpeg')
    smoother_fig.savefig(root+'/smoother_results.jpeg')

    print('\nsaving filter video.....')
    filter_vid.save(root+'/filter_results.mp4', writer = 'ffmpeg', dpi = 200) """

    """ print('making LL video.....')
    LL_fig, LL_vid, _ = probabilistic_model_likelihood(
        model = P_x__y_h, model_name = 'P(X|Y,H)', 
        input_data = observations, label_data = latent, 
        posNorm = posNorm, untransform_label = False,
        grid_dim = 100, plotting = True, video = True,
        )
    print('saving LL video.....')
    LL_vid.save(root+'/test_likelihood.mp4', writer = 'ffmpeg', dpi = 200) """


    """ plot_model_predictions(
        pred_mean = smoother_trace.mean(dim = 0), pred_vars = smoother_trace.var(dim = 0),
        label_data = latent, title = 'Smoother Performance',
        )

    plt.show() """











