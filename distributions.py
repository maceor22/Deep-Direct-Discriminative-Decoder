import torch
from torch.distributions import constraints
#from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal



class GaussianMixture1(object):

    arg_constraints = {
        "pi_k" : constraints.greater_than_eq,
        "loc_k" : constraints.real,
        "scale_k" : constraints.positive,
    }

    def __init__(
            self, pi_k, loc_k, scale_k,
            #validate_args = False,
        ):

        if pi_k.dim() == 1:
            self.n_mixtures = pi_k.size(0)
        elif pi_k.dim() == 2:
            self.n_mixtures = pi_k.size(1)
        self.pi_k = torch.softmax(pi_k.view(-1, self.n_mixtures), dim = 1)

        #batch_shape = self.pi_k.shape[:-1]
        self.event_shape = loc_k.shape[-1:]
        
        self.loc_k = loc_k.view(-1, self.n_mixtures, self.event_shape[0])

        self.scale_k = scale_k.view(-1, self.n_mixtures, self.event_shape[0])
        
        #super().__init__(batch_shape, event_shape, validate_args = validate_args)

    def sample(self):
        with torch.no_grad():
            k = self.pi_k.multinomial(num_samples = 1).squeeze(1)

            loc = torch.stack(
                [self.loc_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            scale = torch.stack(
                [self.scale_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            
            return Normal(loc = loc, scale = scale).sample()
    

    def log_prob(self, value):
        probs = torch.ones((value.size(0)), device = value.device)

        if value.requires_grad:
            probs = probs.requires_grad_()
        
        for i in range(self.event_shape[0]):
            temp = torch.zeros((value.size(0)), device = value.device)
            if value.requires_grad:
                temp = temp.requires_grad_()
            
            for k in range(self.n_mixtures):
                gauss = Normal(loc = self.loc_k[:,k,i], scale = self.scale_k[:,k,i])
                temp = temp + torch.exp(gauss.log_prob(value[:,i])) * self.pi_k[:,k]

            probs = probs * temp

        return torch.log(probs).nan_to_num(neginf = -100)


class GaussianMixture(object):

    arg_constraints = {
        "pi_k" : constraints.greater_than_eq,
        "loc_k" : constraints.real_vector,
        "covariance_matrix_k" : constraints.positive_definite,
        "scale_tril_k" : constraints.lower_cholesky,
    }

    def __init__(
            self, pi_k, loc_k, covariance_type = 'diag',
            variance_k = None, covariance_matrix_k = None, scale_tril_k = None,
            #validate_args = False,
        ):

        self.pi_k = torch.softmax(pi_k, dim = -1)
        self.loc_k = loc_k

        batch_shape = loc_k.shape[:-2]
        event_shape = loc_k.shape[-2:]

        if covariance_type == 'diag':
            scale_tril_k = variance_k**0.5 * torch.eye(event_shape[1]).expand(*[_ for _ in batch_shape], -1, -1)

        elif covariance_type == 'full':
            if covariance_matrix_k is not None:
                scale_tril_k = torch.linalg.cholesky(covariance_matrix_k)
        
        self.scale_tril_k = scale_tril_k
        

    def sample(self):
        with torch.no_grad():
            k = self.pi_k.multinomial(num_samples = 1).squeeze(1)

            loc = torch.stack(
                [self.loc_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            scale_tril = torch.stack(
                [self.scale_tril_k[i,k[i],:,:] for i in range(k.size(0))], dim = 0)
            
            return MultivariateNormal(loc = loc, scale_tril = scale_tril).sample()
    

    def log_prob(self, value):
        prob_k = MultivariateNormal(
            loc = self.loc_k, scale_tril = self.scale_tril_k
        ).log_prob(value.expand(self.loc_k.size(-2),-1,-1).transpose(0,1)).exp()

        probs = (self.pi_k * prob_k).sum(dim = -1)

        return torch.log(probs).nan_to_num(neginf = -100)


class MultivariateNormalMixture(object):

    arg_constraints = {
        "pi_k" : constraints.greater_than_eq,
        "loc_k" : constraints.real_vector,
        "covariance_matrix_k" : constraints.positive_definite,
        "scale_tril_k" : constraints.lower_cholesky,
    }

    def __init__(
            self, pi_k, loc_k, 
            covariance_matrix_k = None, scale_tril_k = None,
            #validate_args = False,
        ):

        if pi_k.dim() == 1:
            self.n_mixtures = pi_k.size(0)
        elif pi_k.dim() == 2:
            self.n_mixtures = pi_k.size(1)
        self.pi_k = torch.softmax(pi_k.view(-1, self.n_mixtures), dim = 1)

        #batch_shape = self.pi_k.shape[:-1]
        event_shape = loc_k.shape[-1:]
        
        self.loc_k = loc_k.view(-1, self.n_mixtures, event_shape[0])

        if covariance_matrix_k is not None:
            scale_tril_k = torch.linalg.cholesky(covariance_matrix_k)
        self.scale_tril_k = scale_tril_k.view(
            -1, self.n_mixtures, event_shape[0], event_shape[0])
        
        #super().__init__(batch_shape, event_shape, validate_args = validate_args)

    def sample(self):
        with torch.no_grad():
            k = self.pi_k.multinomial(num_samples = 1).squeeze(1)

            loc = torch.stack(
                [self.loc_k[i,k[i],:] for i in range(k.size(0))], dim = 0)
            scale_tril = torch.stack(
                [self.scale_tril_k[i,k[i],:,:] for i in range(k.size(0))], dim = 0)
            
            return MultivariateNormal(loc = loc, scale_tril = scale_tril).sample()
    

    def log_prob(self, value):
        probs = torch.zeros((value.size(0)), device = value.device)

        if value.requires_grad:
            probs = probs.requires_grad_()
        
        for k in range(self.n_mixtures):
            mvn = MultivariateNormal(
                loc = self.loc_k[:,k,:], scale_tril = self.scale_tril_k[:,k,:,:])
            temp = torch.exp(mvn.log_prob(value))
            probs = probs + temp*self.pi_k[:,k]

        return torch.log(probs).nan_to_num(neginf = -100)



""" pi_k = torch.ones((10,4))
loc_k = torch.zeros((10,4,2))
scale_tril_k = torch.zeros((10,4,2,2))
scale_tril_k[:,:,0,0] = 1
scale_tril_k[:,:,1,1] = 0.5
scale_k = torch.zeros((10,4,2)).uniform_()**2

#print(loc_k, scale_k)

mvnmix = MultivariateNormalMixture(
    pi_k = pi_k, loc_k = loc_k, scale_tril_k = scale_tril_k)

gaussmix = GaussianMixture(
    pi_k = pi_k, loc_k = loc_k, scale_k = scale_k)

print(gaussmix.sample())

target = torch.zeros((10,2))
target[0,:] = 10000

print(gaussmix.log_prob(target)) """





















