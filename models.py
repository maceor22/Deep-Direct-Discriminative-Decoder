from maze_utils import *
from bayesian import BayesLinear
from ltc import LTC
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Binomial, Normal, MultivariateNormal
import time as time
from memory_profiler import profile


# random walk model
class RandomWalk(nn.Module):
    
    def __init__(self, xvar = 0.2857/200, yvar = 0.6330/200):
        super(RandomWalk, self).__init__()
        
        covar = torch.zeros((2,2))
        covar[0,0] = xvar
        covar[1,1] = yvar
        self.covar = covar

    def forward(self, x, var = None):
        if var is None:
            return MultivariateNormal(x.squeeze(1), self.covar).sample()
        else:
            covar = torch.zeros((2,2))
            covar[0,0] = var[0]
            covar[1,1] = var[1]
            return MultivariateNormal(x, covar).sample()
    
    def predict(self, x, var = None):
        return self.forward(x[:,:,:2], var)


# MLP classifier generating probability that next timestep is in each class
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
            x = torch.argmax(x, dim = 1)
            out = torch.zeros((x.size(0),self.latent_dim), device = x.device)
            for i in range(x.size(0)):
                out[i,x[i]] = 1
            return out
        

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
        return logits
    
    # method returning optionally output arm or log probability values for each arm
    def predict(self, x, return_sample = True):
        logits = self.forward(x)
        
        if return_sample:
            pred = Binomial(total_count = self.n_trials, logits = logits).sample()
        else:
            pred = self.n_trials * logits.exp()
        
        if self.log_target:
            pred = torch.log(pred + 1)
        
        return pred
        
    def prob(self, x, target, reduction = 'mean'):
        if self.log_target:
            target = (target.exp() - 1).long()
        
        logits = self.forward(x)

        log_probs = Binomial(
            total_count = self.n_trials, logits = logits,
            ).log_prob(target).view(-1,self.latent_dim)
        
        if reduction == 'joint':
            probs = log_probs.sum(dim = 1).exp()
        elif reduction == 'mean':
            probs = log_probs.exp().mean(dim = 1)
        
        return probs



# MLP generative model producing prediction distribution of next timestep
class GaussianMLP(nn.Module):
    
    def __init__(
            self, hidden_layer_sizes, 
            input_dim, latent_dim, 
            #epsilon = 1e-40,
            ):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        self.layer_sizes = layer_sizes
        self.latent_dim = latent_dim
        #self.epsilon = epsilon
        self.lin = nn.Sequential(*layers)
        self.mu = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_sigma = nn.Linear(layer_sizes[-1], latent_dim)
    
    def load_parameters(self, params):
        lower = 0
        ix = 0
        for layer in self.lin:
            if type(layer) == nn.modules.linear.Linear:
                upper = lower + self.layer_sizes[ix]*self.layer_sizes[ix+1]
                weight = params[lower:upper].reshape(self.layer_sizes[ix], self.layer_sizes[ix+1])
                
                lower = upper
                upper += self.layer_sizes[ix+1]
                bias = params[lower:upper]
                
                layer.weight = nn.Parameter(weight)
                layer.bias = nn.Parameter(bias)
                
                lower = upper
                ix += 1
        
        upper += self.layer_sizes[-1]*self.latent_dim*2
        weight = params[lower:upper].reshape(self.layer_sizes[-1], self.latent_dim*2)
        bias = params[upper:]
        
        self.final.weight = nn.Parameter(weight)
        self.final.bias = nn.Parameter(bias)
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma(x)) #+ self.epsilon
        return mu, sigma
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        mu, sigma = self.forward(x)
        
        if return_sample:            
            noise = Normal(torch.zeros_like(mu), sigma).sample()
            # return sample
            return mu + noise
        else:
            # return prediction mean and prediction variance
            return mu, sigma
    

    def prob(self, x, y):
        mu, sigma = self.forward(x)

        log_probs = torch.zeros((x.size(0)), requires_grad = True, device = x.device)

        for i in range(self.latent_dim):
            log_probs = log_probs + Normal(mu[:,i], sigma[:,i]).log_prob(y[:,i])
        
        return log_probs.exp()


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
        return mu, sigma
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True):
        mu, sigma = self.forward(x)
        
        if return_sample:            
            noise = Normal(torch.zeros_like(mu), sigma).sample()
            # return sample
            return mu + noise
        else:
            # return prediction mean and prediction variance
            return mu, sigma


# MLP generative model producing prediction distribution of next timestep
class MultivariateNormalMLP(nn.Module):
    
    def __init__(self, hidden_layer_sizes, input_dim, latent_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(MultivariateNormalMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        self.layer_sizes = layer_sizes
        self.latent_dim = latent_dim
        self.lin = nn.Sequential(*layers)
        self.mu = nn.Linear(layer_sizes[-1], latent_dim)
        self.num_tril_components = int(latent_dim*(latent_dim+1)/2)
        self.tril_components = nn.Linear(layer_sizes[-1], self.num_tril_components)
        
    # ...
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
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        mu = self.mu(x)
        cholesky_tril = self.format_cholesky_tril(self.tril_components(x))
        return mu, cholesky_tril
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, scale_tril = False):
        mu, cholesky_tril = self.forward(x)
        
        if return_sample:
            sample = MultivariateNormal(mu, scale_tril = cholesky_tril).sample()
            if mu.requires_grad:
                return sample.requires_grad_()
            else:
                return sample
        else:
            # return prediction mean and prediction variance
            if scale_tril:
                return mu, cholesky_tril
            else:
                return mu, torch.matmul(cholesky_tril, cholesky_tril.transpose(1,2))


# LSTM generative model producing X-Y prediction distribution of next timestep
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


# p = ((Lin - 1) * stride - Lin + kernel) / 2
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
        


class GaussianLTC(nn.Module):

    def __init__(
            self, hidden_layer_size, num_layers,
            sequence_dim, feature_dim, latent_dim,
            last_hidden_state = False,
            use_cell_memory = False, solver_unfolds = 6, 
            ):
        
        super(GaussianLTC, self).__init__()

        self.ltc = LTC(
            input_dim = feature_dim, hidden_dim = hidden_layer_size, num_layers = num_layers,
            use_cell_memory = use_cell_memory, ode_solver_unfolds = solver_unfolds,
            )
        
        self.mu = nn.Linear(
            in_features = hidden_layer_size if last_hidden_state else sequence_dim*hidden_layer_size, 
            out_features = latent_dim,
            )
        self.log_sigma = nn.Linear(
            in_features = hidden_layer_size if last_hidden_state else sequence_dim*hidden_layer_size, 
            out_features = latent_dim,
            )
        
        self.seq_dim = sequence_dim
        self.hid_dim = hidden_layer_size
        self.last_hidden = last_hidden_state

    
    def forward(self, x):
        hidden_state = self.ltc(x)[0].view(-1, self.seq_dim, self.hid_dim)

        if self.last_hidden:
            hidden_state = hidden_state[:,-1,:]
        else:
            hidden_state = hidden_state.flatten(1,-1)            

        mu = self.mu(hidden_state)
        sigma = torch.exp(self.log_sigma(hidden_state))

        return mu, sigma
    

    def predict(self, x, return_sample = True):
        mu, sigma = self.forward(x)

        if return_sample:
            return Normal(mu, sigma).sample()
        else:
            return mu, sigma


class MultivariateNormalMixture(object):
    
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
        return g, nll
                



# MLP gaussian mixture model producing prediction distribution of next timestep
class GaussianMixtureMLP(nn.Module):
    
    def __init__(self, hidden_layer_sizes, num_mixtures, input_dim, latent_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(GaussianMixtureMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        self.layer_sizes = layer_sizes
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim
        self.lin = nn.Sequential(*layers)
        
        self.pi = nn.Sequential(
            nn.Linear(layer_sizes[-1], num_mixtures),
            nn.Softmax(dim = 1),
            )
        self.mu = nn.Linear(layer_sizes[-1], num_mixtures*latent_dim)
        self.log_sigma = nn.Linear(layer_sizes[-1], num_mixtures*latent_dim)        
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        pi_k = self.pi(x).view(-1, self.num_mixtures)
        mu_k = self.mu(x).view(-1, self.num_mixtures, self.latent_dim).nan_to_num(nan = torch.rand(1).item())
        sigma_k = torch.exp(self.log_sigma(x).view(-1, self.num_mixtures, self.latent_dim)).nan_to_num(nan = torch.rand(1).item())
        return pi_k, mu_k, sigma_k
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, method = 'sample'):
        pi_k, mu_k, sigma_k = self.forward(x)

        if method == 'sample':
            k = pi_k.multinomial(num_samples = 1).squeeze(1)
        
        elif method == 'dominant':
            k = pi_k.argmax(dim = 1)

        if method == 'sample' or method == 'dominant':
            mu = torch.stack([mu_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            sigma = torch.stack([sigma_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
        
        elif method == 'average':
            for i in range(x.size(0)):
                for k in range(self.num_mixtures):
                    mu_k[i,k,:] *= pi_k[i,k]
                    sigma_k[i,k,:] *= pi_k[i,k]
            mu = mu_k.sum(dim = 1)
            sigma = sigma_k.sum(dim = 1)
        
        if return_sample: 
            if mu.size(0) < 2000:
                samples = Normal(mu, sigma).sample()
            else:
                samples = []
                lower = 0
                done = False
                while not done:
                    upper = lower + 2000
                    if upper > mu.size(0):
                        upper = mu.size(0)
                    samples.append(Normal(mu[lower:upper], sigma[lower:upper]).sample())
                    if upper == mu.size(0):
                        done = True
                    lower += 2000
                samples = torch.cat(samples, dim = 0)
            # return samples
            return samples
        else:
            # return prediction mean and prediction variance
            return mu, sigma
        
        
    def prob(self, x, target):
        pi_k, mu_k, sigma_k = self.forward(x)

        probs = torch.ones((x.size(0)), requires_grad = True, device = x.device)

        for i in range(self.latent_dim):
            temp = torch.zeros((x.size(0)), requires_grad = True, device = x.device)
            for k in range(self.num_mixtures):
                gauss = Normal(loc = mu_k[:,k,i], scale = sigma_k[:,k,i])
                temp += (torch.exp(gauss.log_prob(target[:,i])) * pi_k[:,k])

            probs *= temp
                    
        return probs
        

# MLP gaussian mixture model producing prediction distribution of next timestep
class MultivariateNormalMixtureMLP(nn.Module):
    
    def __init__(self, hidden_layer_sizes, num_mixtures, input_dim, latent_dim):
        # hidden_layer_sizes: list or tuple containing hidden layer sizes
        # input_dim: dimension of input to the model
        
        super(MultivariateNormalMixtureMLP, self).__init__()
        
        layer_sizes = hidden_layer_sizes
        layer_sizes.insert(0, input_dim)
        
        # build hidden layers with ReLU activation function in between
        layers = []
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        self.layer_sizes = layer_sizes
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim
        self.lin = nn.Sequential(*layers)
        
        self.pi = nn.Sequential(
            nn.Linear(layer_sizes[-1], num_mixtures),
            nn.Softmax(dim = 1),
            )
        self.mu = nn.Linear(layer_sizes[-1], num_mixtures*latent_dim)
        self.num_tril_components = int(latent_dim*(latent_dim+1)/2)
        self.tril_components = nn.Linear(layer_sizes[-1], num_mixtures*self.num_tril_components)        
    
    # ...
    def format_cholesky_tril(self, tril_components):
        tril_components = tril_components.view(-1, self.num_mixtures, self.num_tril_components)
        
        cholesky_tril = torch.zeros((
            tril_components.size(0), self.num_mixtures, self.latent_dim, self.latent_dim,
            ), device = tril_components.device)
        for k in range(self.num_mixtures):
            idx = 0
            for i in range(self.latent_dim):
                for j in range(i+1):
                    if i == j:
                        cholesky_tril[:,k,i,j] = torch.exp(tril_components[:,k,idx])
                    else:
                        cholesky_tril[:,k,i,j] = tril_components[:,k,idx]
                    idx += 1
        
        return cholesky_tril
    
    # TBD
    def load_parameters(self, params):
        pass
    
    # forward call
    def forward(self, x):
        x = self.lin(x.flatten(1,-1))
        pi_k = self.pi(x).view(-1, self.num_mixtures)
        mu_k = self.mu(x).view(-1, self.num_mixtures, self.latent_dim)
        cholesky_tril_k = self.format_cholesky_tril(self.tril_components(x))
        return pi_k, mu_k, cholesky_tril_k
        
    # method for optionally producing prediction distribution or sample from distribution
    def predict(self, x, return_sample = True, scale_tril = False, method = 'sample'):
        pi_k, mu_k, cholesky_tril_k = self.forward(x)

        if method == 'sample':
            k = pi_k.multinomial(num_samples = 1).squeeze(1)
        
        elif method == 'dominant':
            k = pi_k.argmax(dim = 1)

        if method == 'sample' or method == 'dominant':
            mu = torch.stack([mu_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            cholesky = torch.stack([cholesky_tril_k[i,k[i],:,:] for i in range(k.size(0))], dim = 0)
            tril = True

        elif method == 'average':
            sigma_k = cholesky_tril_k @ cholesky_tril_k.transpose(2,3)
            for i in range(x.size(0)):
                for k in range(self.num_mixtures):
                    mu_k[i,k,:] *= pi_k[i,k]
                    sigma_k[i,k,:,:] *= pi_k[i,k]
            mu = mu_k.sum(dim = 1)
            sigma = sigma_k.sum(dim = 1)
            tril = False
        
        if return_sample: 
            if mu.size(0) < 2000:
                if tril:
                    samples = MultivariateNormal(loc = mu, scale_tril = cholesky).sample()
                else:
                    samples = MultivariateNormal(loc = mu, covariance_matrix = sigma).sample()
            else:
                samples = []
                lower = 0
                done = False
                while not done:
                    upper = lower + 2000
                    if upper > mu.size(0):
                        upper = mu.size(0)
                    if tril:
                        samps = MultivariateNormal(
                            loc = mu[lower:upper], scale_tril = cholesky[lower:upper]).sample()
                    else:
                        samps = MultivariateNormal(
                            loc = mu[lower:upper], covariance_matrix = sigma[lower:upper]).sample()
                    samples.append(samps)
                    if upper == mu.size(0):
                        done = True
                    lower += 2000
                samples = torch.cat(samples, dim = 0)
            # return samples
            return samples
        else:
            if scale_tril:
                if not tril:
                    cholesky = torch.linalg.cholesky(sigma)
                return mu, cholesky
            else:
                # return prediction mean and prediction variance
                return mu, sigma
        
    #@profile  
    def prob(self, x, target):        
        pi_k, mu_k, cholesky_tril_k = self.forward(x)
        
        probs = torch.zeros((x.size(0)), requires_grad = True, device = x.device)
        
        for k in range(self.num_mixtures):
            mvn = MultivariateNormal(
                loc = mu_k[:,k,:], scale_tril = cholesky_tril_k[:,k,:,:])
            temp = torch.exp(mvn.log_prob(target.float()))
            probs = probs + temp*pi_k[:,k]
        
        return probs
    

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
                






