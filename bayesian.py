from maze_utils import *
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import time as time



class BayesLinear(nn.Module):

    def __init__(self, in_features, out_features, distribution = 'gaussian'):

        super(BayesLinear, self).__init__()

        if distribution == 'gaussian':
            w_param1 = torch.zeros((in_features, out_features)).uniform_(-1e-2,1e-2)
            logw_param2 = torch.zeros((in_features, out_features)).uniform_(-4,-2)
            
            b_param1 = torch.zeros((out_features,)).uniform_(-1e-2,1e-2)
            logb_param2 = torch.zeros((out_features,)).uniform_(-4,-2)

            self.w_param1 = nn.Parameter(w_param1, requires_grad = True)
            self.logw_param2 = nn.Parameter(logw_param2, requires_grad = True)
            
            self.b_param1 = nn.Parameter(b_param1, requires_grad = True)
            self.logb_param2 = nn.Parameter(logb_param2, requires_grad = True)
            
            self.distribution = 'gaussian'
    

    def forward(self, x):
        N = x.size(0)
        device = x.device

        if self.distribution == 'gaussian':
            w_distribution = Normal(self.w_param1, torch.exp(self.logw_param2))
            b_distribution = Normal(self.b_param1, torch.exp(self.logb_param2))

        weights = w_distribution.sample((N,)).to(device).requires_grad_()
        biases = b_distribution.sample((N,)).to(device).requires_grad_()


        
        return (x.unsqueeze(1) @ weights).squeeze(1) + biases


    def load_parameters(
            self, weight_param1, weight_param2, bias_param1, bias_param2,
            ):
        if self.distribution == 'gaussian':
            self.w_param1 = nn.Parameter(weight_param1)
            self.logw_param2 = nn.Parameter(weight_param2)
            self.b_param1 = nn.Parameter(bias_param1)
            self.logb_param2 = nn.Parameter(bias_param2)


















