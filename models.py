from maze_utils import *
from bayesian import BayesLinear
from ltc import LTC
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import *
import time as time
#from memory_profiler import profile

""" 
class WishartWrapper(nn.Module):

    def __init__(
            self, model, df, prior,
            model_weight = 1, wishart_weight = 1,
            ):
        super(WishartWrapper, self).__init__()

        self.model = model
        self.df = df
        self.prior = prior
        self.model_w = model_weight / (model_weight + wishart_weight)
        self.wishart_w = wishart_weight / (model_weight + wishart_weight)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x, return_sample = True):
        distribution = self.model(x)

        if return_sample:
            return distribution.sample()
        else:
            return distribution
        
    def log_prob(self, x, y):
        if self.model_w > 0:
            model_ll = self.model.log_prob(x,y)

        distribution = self.model(x)

        if type(distribution) == Normal:
            I = torch.eye(self.prior.size(0))

            scale_tril = self.prior * I
            scale_tril = scale_tril.expand(x.size(0),-1,-1).to(x.device)

            value = distribution.scale.unsqueeze(-1).repeat(1,1,self.prior.size(0))
            value = value * I.expand(x.size(0),-1,-1)

            wishart_ll = Wishart(
                df = self.df.expand(x.size(0)).to(x.device),
                scale_tril = scale_tril,
            ).log_prob(value)

        elif type(distribution) == MultivariateNormal:
            scale_tril = torch.linalg.cholesky(self.prior)
            scale_tril = scale_tril.expand(x.size(0),-1,-1).to(x.device)

            value = distribution.scale_tril @ distribution.scale_tril.transpose(-2,-1)

            wishart_ll = Wishart(
                df = self.df.expand(x.size(0)).to(x.device),
                scale_tril = scale_tril,
            ).log_prob(value)

        elif type(distribution) == MixtureSameFamily:
            scale_tril = torch.linalg.cholesky(self.prior)
            scale_tril = scale_tril.expand(x.size(0),-1,-1,-1).to(x.device)

            value = distribution.component_distribution.base_dist.scale_tril
            value = value @ value.transpose(-2,-1)

            wishart_ll = Wishart(
                df = self.df.expand(x.size(0),-1).to(x.device),
                scale_tril = scale_tril,
            ).log_prob(value).mean(dim = -1)
        
        
        if self.model_w > 0:
            #print(model_ll.mean(0).item(), wishart_ll.mean(0).item())
            return self.model_w*model_ll + self.wishart_w*wishart_ll
        else:
            #print(wishart_ll.mean(0).item())
            return wishart_ll
 """

""" # random walk model
class GaussianTransition(nn.Module):
    
    def __init__(self, sigma):
        super(GaussianTransition, self).__init__()
        
        self.sigma = sigma

    def forward(self, x):
        if x.dim() == 3:
            x = x[:,-1,:]
        return Normal(x, self.sigma.to(x.device).expand(x.size(0),-1))
        
    def predict(self, x, return_sample = True):
        gauss = self.forward(x)
        
        if return_sample:
            return gauss.sample()
        else:
            return gauss
    
    def prob(self, x, y):
        gauss = self.forward(x)
        return gauss.log_prob(y).sum(dim = -1).exp()
 """

""" 
class GaussianTransitionLinear(nn.Module):
    
    def __init__(
            self, input_dim, latent_dim, sigma_init = None,
            ):
        super(GaussianTransitionLinear, self).__init__()
        
        self.latent_dim = latent_dim
        self.log_sigma = nn.Linear(input_dim, latent_dim)

        if sigma_init == None:
            self.log_sigma.bias = nn.Parameter(
                torch.log(sigma_init).requires_grad_())
        
        self.log_sigma.weight = nn.Parameter(
            torch.zeros_like(self.log_sigma.weight))
        self.log_sigma.weight.requires_grad_()

    def forward(self, x):
        sigma = self.log_sigma(x.flatten(-2,-1)).exp()
        if x.dim() == 3:
            return Normal(x[:,-1,:], sigma)
        elif x.dim() == 4:
            return Normal(x[:,:,-1,:], sigma)
        
    def predict(self, x, return_sample = True):
        gauss = self.forward(x)
        
        if return_sample:
            return gauss.sample()
        else:
            return gauss
    
    def prob(self, x, y):
        gauss = self.forward(x)
        return gauss.log_prob(y).sum(dim = -1).exp()
 """

""" 
class GaussianTransitionMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            ):
        super(GaussianTransitionMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)

        
        # build hidden layers
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], latent_dim))
        
        self.latent_dim = latent_dim
        self.log_sigma = nn.Sequential(*layers)

    def forward(self, x):
        sigma = self.log_sigma(x.flatten(-2,-1)).exp()
        return Normal(x[...,-1,:], sigma)
        
    def predict(self, x, return_sample = True):
        gauss = self.forward(x)
        
        if return_sample:
            return gauss.sample()
        else:
            return gauss
    
    def log_prob(self, x, y):
        gauss = self.forward(x)
        return gauss.log_prob(y).sum(dim = -1)
 """

""" 
class MultivariateNormalTransitionMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            activation = 'relu',
            ):
        super(MultivariateNormalTransitionMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)

        if activation == 'relu':
            activ = nn.Relu()
        elif activation == 'tanh':
            activ = nn.Tanh()
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(activ)

        self.num_tril_components = int(latent_dim*(latent_dim+1)/2)
        layers.append(nn.Linear(layer_sizes[-1], self.num_tril_components))
        
        self.latent_dim = latent_dim
        self.tril_components = nn.Sequential(*layers)

    def format_cholesky_tril(self, tril_components):
        tril_components = tril_components.view(-1, self.num_tril_components)
        
        cholesky_tril = torch.zeros(
            (tril_components.size(0), self.latent_dim, self.latent_dim),
            device = tril_components.device,
            )
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i+1):
                if i == j:
                    cholesky_tril[:,i,j] = torch.exp(tril_components[:,idx])
                else:
                    cholesky_tril[:,i,j] = tril_components[:,idx]
                idx += 1
        
        return cholesky_tril
    
    def forward(self, x):
        tril_components = self.tril_components(x.flatten(1,-1))
        cholesky_tril = self.format_cholesky_tril(tril_components)
        return MultivariateNormal(loc = x[:,-1,:], scale_tril = cholesky_tril)
        
    def predict(self, x, return_sample = True):
        mvn = self.forward(x)
        
        if return_sample:
            return mvn.sample()
        else:
            return mvn
    
    def prob(self, x, y):
        mvn = self.forward(x)
        return mvn.log_prob(y).exp()
 """

""" 
class ClassifierMLP(nn.Module):
    
    def __init__(self, hidden_layer_sizes, input_dim, latent_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # num_arms: number of arms in discrete transform
        # input_dim: input dimension to model
        
        super(ClassifierMLP, self).__init__()
        
        self.latent_dim = latent_dim
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_dim))
        
        self.lin = nn.Sequential(*layers)
        self.final = nn.LogSoftmax(dim = 1)
        
    # forward call
    def forward(self, x):
        return self.final(self.lin(x.flatten(1,-1)))
    
    # method returning optionally output arm or log probability values for each arm
    def predict(self, x, return_sample = True):
        if not return_sample:
            return self.forward(x)
        else:
            x = self.forward(x)
            return torch.argmax(x, dim = -1)
 """  

class BinomialMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            n_trials, log_target = False,
        ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # num_arms: number of arms in discrete transform
        # input_dim: input dimension to model
        
        super(BinomialMLP, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_trials = n_trials
        self.log_target = log_target
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_dim))
        
        self.lin = nn.Sequential(*layers)
        self.final = nn.LogSigmoid()
        
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        logits = self.final(x).nan_to_num(nan = -10, posinf = -10, neginf = -10)
        binomial = Binomial(total_count = self.n_trials, logits = logits)
        return binomial
    
    def predict(self, x, return_sample = True):
        binomial = self.forward(x)
        
        if return_sample:
            return binomial.sample()
        else:
            return binomial
                
    def log_prob(self, x, y):
        binomial = self.forward(x)        
        return binomial.log_prob(y).sum(dim = -1)



class GaussianMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            covariance_type = 'diag', dropout_p = 0.5, epsilon = 1e-20,
            ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Dropout(p = dropout_p))
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        

        self.lin = nn.Sequential(*layers)
        self.mu = nn.Sequential(
            nn.Dropout(p = dropout_p), nn.Linear(layer_sizes[-1], latent_dim))

        if covariance_type == 'diag':
            num_tril_components = latent_dim
        elif covariance_type == 'full':
            num_tril_components = int(latent_dim*(latent_dim+1)/2)
        
        self.tril_components = nn.Sequential(
            nn.Dropout(p = dropout_p), nn.Linear(layer_sizes[-1], num_tril_components))
        
        self.latent_dim = latent_dim
        self.covar_type = covariance_type
        self.epsilon = epsilon

    # ...
    def format_cholesky_tril(self, tril_components):
        batch_shape = tril_components.shape[:-1]
        cholesky_tril = torch.zeros(
            (*[_ for _ in batch_shape], self.latent_dim, self.latent_dim),
            device = tril_components.device,
            )
        
        if self.covar_type == 'diag':
            for i in range(self.latent_dim):
                cholesky_tril[...,i,i] = tril_components[...,i].exp() + self.epsilon

        elif self.covar_type == 'full':
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        cholesky_tril[...,i,j] = tril_components[...,idx].exp() + self.epsilon
                    else:
                        cholesky_tril[...,i,j] = tril_components[...,idx]
                    idx += 1
        
        return cholesky_tril
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(-2,-1))
        mu = self.mu(x)
        cholesky_tril = self.format_cholesky_tril(self.tril_components(x))
        return MultivariateNormal(loc = mu, scale_tril = cholesky_tril)
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        mvn = self.forward(x)
        
        if return_sample:
            return mvn.sample()
        else:
            return mvn
        
    
    def log_prob(self, x, y):
        mvn = self.forward(x)
        return mvn.log_prob(y)


class GaussianBayesMLP(nn.Module):

    def __init__(
            self, hidden_layer_sizes, input_dim, latent_dim,
            parameter_distribution = 'gaussian',
        ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianBayesMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(BayesLinear(
                layer_sizes[i-1], layer_sizes[i], parameter_distribution,
            ))
            layers.append(nn.ReLU())
        
        self.layer_sizes = layer_sizes
        self.latent_dim = latent_dim
        self.lin = nn.Sequential(*layers)
        self.mu = BayesLinear(layer_sizes[-1], latent_dim, parameter_distribution)
        self.log_sigma = BayesLinear(layer_sizes[-1], latent_dim, parameter_distribution)
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma(x))
        return Normal(loc = mu, scale = sigma)
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        normal = self.forward(x)
        
        if return_sample:            
            # return sample
            return normal.sample()
        else:
            # return prediction mean and prediction variance
            return normal



class GaussianLSTM(nn.Module):
    
    def __init__(self, hidden_layer_size, num_layers, sequence_dim, feature_dim, latent_dim):
        # hidden_layer_sizes: integer dictating size of hidden LSTM layers
        # num_layers: integer dictating number of LSTM layers
        # sequence_dim: length of input sequence dimension
        # feature_dim: number of features in input
        # latent_dim: size of output dimension
        
        super(GaussianLSTM, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(
            feature_dim, hidden_layer_size, 
            num_layers = num_layers, batch_first = True)
        
        self.final = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(hidden_layer_size*sequence_dim, latent_dim),
            )
        
    # forward call
    def forward(self, x):
        return self.final(self.lstm(x)[0]).squeeze()
    
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        out = self.forward(x)
        if return_sample:
            out[:,:self.latent_dim] += Normal(
                torch.zeros((out.size(0), self.latent_dim)), 
                torch.abs(out[:,self.latent_dim:])).sample()
            # return sample
            return out[:,self.latent_dim:]
        else:
            # return prediction mean and prediction variance
            return out[:,:self.latent_dim], torch.abs(out[:,self.latent_dim:])


""" # p = ((Lin - 1) * stride - Lin + kernel) / 2
class GaussianCNN(nn.Module):
    
    def __init__(self, 
                 hidden_layer_sizes, kernel_sizes, stride_sizes, 
                 sequence_dim, feature_dim, latent_dim,
                 ):
        
        super(GaussianCNN, self).__init__()
        
        self.sequence_dim = sequence_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, feature_dim)
        
        layers = []
        for i in range(1,len(layer_sizes)):
            pad = int(((sequence_dim - 1) * stride_sizes[i-1] - sequence_dim + kernel_sizes[i-1]) / 2)
            layers.append(nn.Conv1d(
                in_channels = layer_sizes[i-1], out_channels = layer_sizes[i], 
                kernel_size = kernel_sizes[i-1], stride = stride_sizes[i-1], padding = pad,
                ))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(
                kernel_size = kernel_sizes[i-1], stride = stride_sizes[i-1], padding = pad,
                ))
            
        self.cnn = nn.Sequential(*layers)
        self.flat = nn.Flatten()
        self.final = nn.Linear(sequence_dim*layer_sizes[-1], latent_dim*2)
        
    
    def forward(self, x):
        return self.final(self.flat(self.cnn(x.transpose(1,2))))
    
    
    def predict(self, x, return_sample = True):
        out = self.forward(x)
        if return_sample:
            out[:,:self.latent_dim] += Normal(
                torch.zeros((out.size(0), self.latent_dim)), 
                torch.abs(out[:,self.latent_dim:])).sample()
            # return sample
            return out[:,self.latent_dim:]
        else:
            # return prediction mean and prediction variance
            return out[:,:self.latent_dim], torch.abs(out[:,self.latent_dim:])
 """        


class GaussianLTC(nn.Module):

    def __init__(
            self, hidden_layer_size, num_layers,
            sequence_dim, feature_dim, latent_dim,
            use_cell_memory = False, solver_unfolds = 6, 
            ):
        
        super(GaussianLTC, self).__init__()

        self.ltc = LTC(
            input_dim = feature_dim, hidden_dim = hidden_layer_size, num_layers = num_layers,
            use_cell_memory = use_cell_memory, ode_solver_unfolds = solver_unfolds,
            )
        
        self.mu = nn.Linear(
            in_features = hidden_layer_size, out_features = latent_dim,
            )
        self.log_sigma = nn.Linear(
            in_features = hidden_layer_size, out_features = latent_dim,
            )
        
        self.seq_dim = sequence_dim
        self.hid_dim = hidden_layer_size

    
    def forward(self, x):
        hidden_state = self.ltc(x)[0].view(-1, self.seq_dim, self.hid_dim)[:,-1,:]

        mu = self.mu(hidden_state)
        sigma = torch.exp(self.log_sigma(hidden_state))

        return Normal(loc = mu, scale = sigma)
    

    def predict(self, x, return_sample = True):
        normal = self.forward(x)

        if return_sample:
            return normal.sample()
        else:
            return normal


""" class MultivariateNormalMixture(object):
    
    def __init__(
            self, num_mixtures, dim, 
            pi_init = None, mu_init = None, sigma_init = None,
            ):
        
        self.num_mixtures = num_mixtures
        self.dim = dim
        
        if pi_init == None:
            pi_init = torch.zeros((num_mixtures,))
            prob = 1/num_mixtures
            for i in range(pi_init.size(0)):
                pi_init[i] = prob
                
        if mu_init == None:
            mu_init = torch.zeros((num_mixtures, dim)).uniform_(0, 200)
        
        if sigma_init == None:
            sigma_init = torch.zeros((num_mixtures, dim, dim))
            for i in range(dim):
                sigma_init[:,i,i].uniform_(10,20)
        
        self.update_(pi_init, mu_init, sigma_init)
    
    
    def update_(self, new_pi, new_mu, new_sigma):
        self.pi_k = new_pi
        self.mu_k = new_mu
        self.sigma_k = new_sigma
        self.mvn = MultivariateNormal(self.mu_k, self.sigma_k)
        
    

    def fit(self, data, max_iter = 100, threshold = 1e-3, plot_loss = False):
        self.x_n = data.float()
        self.N = data.size(0)
        
        nlls = []
        delta_norms = []
        
        for _ in range(max_iter):    
            g, nll = self.gamma()
            
            sum_g = g.sum(dim = 0)
            
            next_pi = sum_g / self.N
            
            next_mu = torch.matmul(g.transpose(0,1), self.x_n)
            for k in range(self.num_mixtures):
                next_mu[k,:] /= sum_g[k]
            
            next_sigma = torch.zeros((self.num_mixtures, self.dim, self.dim))
            for k in range(self.num_mixtures):
                temp = torch.zeros((self.N, self.dim, self.dim))
                for n in range(self.N):
                    err = self.x_n[n,:] - self.mu_k[k,:]
                    temp[n,:,:] = torch.matmul(err.unsqueeze(1), err.unsqueeze(0))
                    temp[n,:,:] *= g[n,k]
                next_sigma[k,:,:] = temp.sum(dim = 0) / sum_g[k]
                
            delta = torch.cat([
                torch.abs(self.pi_k - next_pi),
                torch.abs(self.mu_k - next_mu).flatten(0,-1),
                torch.abs(self.sigma_k - next_sigma).flatten(0,-1),
                ], dim = 0)
            inf_norm = delta.max(dim = 0)[0].item()
            
            nlls.append(nll)
            delta_norms.append(inf_norm)
            
            self.update_(next_pi, next_mu, next_sigma)
        
            if inf_norm < threshold:
                break
        
        if plot_loss:
            fig, ax = plt.subplots(2, sharex = True)
            
            fig.suptitle('Fitting Density Mixture Model with Expectation-Maximization')
            plt.xlabel('Iterations')
            
            ax[0].plot(range(1,len(nlls)+1), nlls, 'k')
            ax[0].set_ylabel('negative\nlog-likelihood')
            
            ax[1].plot(range(1,len(delta_norms)+1), delta_norms, 'k')
            ax[1].set_ylabel('delta theta\ninf norm')
            
    
    def gamma(self):
        g = torch.zeros((self.N, self.num_mixtures))
        nll = 0
        for n in range(self.N):
            g[n,:] = torch.exp(self.mvn.log_prob(self.x_n[n,:])) * self.pi_k
            temp = g[n,:].sum(dim = 0)
            nll += -torch.log(temp)
            g[n,:] /= temp
        nll /= self.N
        return g, nll """
     

# MLP gaussian mixture model producing prediction distribution of next timestep
class GaussianMixtureMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, num_mixtures, 
            input_dim, latent_dim, covariance_type = 'diag', 
            dropout_p = 0,
            ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianMixtureMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Dropout(p = dropout_p))
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())

        self.lin = nn.Sequential(*layers)
        
        self.pi = nn.Sequential(
            nn.Dropout(p = dropout_p),
            nn.Linear(layer_sizes[-1], num_mixtures),
            nn.Softmax(dim = 1),
            )
        self.mu = nn.Sequential(
            nn.Dropout(p = dropout_p), nn.Linear(layer_sizes[-1], num_mixtures*latent_dim))

        if covariance_type == 'diag':
            num_tril_components = latent_dim
        elif covariance_type == 'full':
            num_tril_components = int(latent_dim*(latent_dim+1)/2)
        
        self.tril_components = nn.Sequential(
            nn.Dropout(p = dropout_p),
            nn.Linear(layer_sizes[-1], num_mixtures*num_tril_components),
            )        

        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim
        self.covar_type = covariance_type

    # ...
    def format_cholesky_tril(self, tril_components):
        batch_shape = tril_components.shape[:-1]
        tril_components = tril_components.view(
            *[_ for _ in batch_shape], self.num_mixtures, -1)
        
        if self.covar_type == 'diag':
            cholesky_tril = tril_components.exp() * torch.eye(self.latent_dim).expand(
                *[_ for _ in batch_shape],self.num_mixtures,-1,-1)
        
        elif self.covar_type == 'full':
            cholesky_tril = torch.zeros((
                *[_ for _ in batch_shape], self.num_mixtures, self.latent_dim, self.latent_dim,
                ), device = tril_components.device)
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        cholesky_tril[...,i,j] = tril_components[...,idx].exp()
                    else:
                        cholesky_tril[...,i,j] = tril_components[...,idx]
                    idx += 1
        
        return cholesky_tril
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        batch_shape = x.shape[:-2]
        x = self.lin(x.flatten(-2,-1))
        mu_k = self.mu(x).view(*[_ for _ in batch_shape], self.num_mixtures, self.latent_dim)
        cholesky_tril_k = self.format_cholesky_tril(self.tril_components(x))

        mix = Categorical(probs = self.pi(x))
        comp = Independent(
            MultivariateNormal(loc = mu_k, scale_tril = cholesky_tril_k), 
            reinterpreted_batch_ndims = 0,
            )
        return MixtureSameFamily(mix, comp)
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        gmm = self.forward(x)
        
        if return_sample: 
            # return samples
            return gmm.sample()
        else:
            # return prediction mean and prediction variance
            return gmm
        
    #@profile  
    def log_prob(self, x, target):
        gmm = self.forward(x)
        return gmm.log_prob(target)

   

# wrapper class for two-model framework
class TwoModelWrapper(object):
    
    def __init__(
            self, modelA, modelB, 
            transform = None,
            modelA_multi_output = True,
            modelB_multi_output = True,
            concat_first = False,
            ):
        # modelA: first model in two model framework
        # modelB: second model in two model framework
        # modelA_multi_output: boolean indicating whether modelA outputs mean and variance
        # modelB_multi_output: boolean indicating whether modelB outputs mean and variance
        # transform: fitted transform object; optionally passed
        
        self.modelA = modelA
        self.modelB = modelB
        self.transform = transform
        self.modelA_multi_output = modelA_multi_output
        self.modelB_multi_output = modelB_multi_output
        self.concat_first = concat_first
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, untransform_data = False):
        # x: input
        # return_sample: boolean indicating whether to return sample
        # untransform_data: boolean indicating whether to untransform 
        
        if self.modelA_multi_output and self.modelB_multi_output:
            pred_meanA, pred_varsA = self.modelA.predict(x, return_sample = False)
            
            if pred_meanA.dim() != 2:
                pred_meanA, pred_varsA = pred_meanA.unsqueeze(1), pred_varsA.unsqueeze(1)
            
            xB = torch.cat([x, pred_meanA, pred_varsA.flatten(1,-1)], dim = 1)
            
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
            return self.transform.untransform(pred_data)
        
        elif not return_sample and untransform_data:
            untf_pred_data, untf_pred_vars = self.transform.untransform(
                pred_data, pred_vars)
            return untf_pred_data, untf_pred_vars
        
        elif not return_sample and not untransform_data:
            return pred_data, pred_vars
        
        else:
            return pred_data


class ModelsWrapper(object):
    
    def __init__(self, models, models_classifier_bool, 
                 transform = None, concat_first = False,
                 ):
        # modelA: first model in two model framework
        # modelB: second model in two model framework
        # modelA_multi_output: boolean indicating whether modelA outputs mean and variance
        # modelB_multi_output: boolean indicating whether modelB outputs mean and variance
        # transform: fitted transform object; optionally passed
        
        self.models = models
        self.class_bool = models_classifier_bool
        self.transform = transform
        self.concat_first = concat_first
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, untransform_data = False):
        # x: input
        # return_sample: boolean indicating whether to return sample
        # untransform_data: boolean indicating whether to untransform 
        
        if return_sample: 
            preds = [] 
        else: 
            pred_params1, pred_params2 = [], []
        
        for i in range(len(self.models)):
            model = self.models[i]
            
            if self.class_bool[i]:
                pred = model.predict(x, return_log_probs = True)
                if pred.dim() != 2:
                    pred = pred.unsqueeze(1)
            else:
                pred_param1, pred_param2 = model.predict(x, return_sample = False)
                if pred_param1.dim() != 2:
                    pred_param1, pred_param2 = pred_param1.unsqueeze(1), pred_param2.unsqueeze(1)
                pred = torch.cat([pred_param1, pred_param2], dim = 1)
                
            preds.append(pred)
        
        ret = torch.cat(
            preds.reverse() if self.concat_first else preds, dim = 1,
            )
        
        if untransform_data:
            ret = self.transform.untransform(ret)
                






