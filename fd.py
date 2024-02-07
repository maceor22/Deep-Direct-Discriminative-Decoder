import torch
from torch import nn as nn


# READ ME !!!!

# FUTURE WORK:

# This implementation is not yet in exact alignment with biological neurons.
#   EPSC from the previous network layer is directly passed as injected 
#     calcium current for the next layer. 
#   The AxonLinear implementation below handles this.
# Two potential approaches to fix this (still in ideation phase) are detailed below.

# DETERMINISTIC APPROACH
#   Dynamics describing how total EPSC arriving at a presynaptic terminal modulates 
#     activity of voltage-gated calcium channels (VGCCs) must be emulated.
#   Activity of VGCCs are obviously voltage-dependent processes.
#     Necessary conversions from current to voltage may have to be made.
#     Resistance of axons may have to be quanitfied.

# PROBABILISTIC APPROACH
#   The Lee et al. (2009) equations suggest that injected calcium current
#     is a function of action potentials received at the presynaptic terminal.
#   This motivates an approach where sequence inputs are converted to spike trains 
#     (produced by Poisson or Binomial count processes) which then propagate through 
#     the facilitation-depression network.
#   A method for detecting spikes from emitted EPSC must then be implemented.
#   Theoretically, a sufficient number of ODE solver steps will result in negligible 
#     differences between network outputs when repeatedly given the same input.


# implementation of facilitation-depression network
class FD(nn.Module):
    
    def __init__(
            self, input_dim, hidden_dims, 
            ode_solver_steps = 6, output_mapping = 'identity',
            ):
        # input_dim : dimensionality of input data
        # hidden_dims : tuple or list indicating number of neurons per network layer
        # ode_solver_steps : number of steps used in numerical differential equation solver
        # output_mapping : string denoting which method to use for output mapping
        super(FD, self).__init__()

        layers = []
        # initialize mapping from input to injected calcium current
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        # constrain injected calcium current to be positive
        layers.append(nn.ReLU())
        # initialize first network layer
        layers.append(FDLayer(hidden_dims[0], ode_solver_steps))

        # if many layers specified
        if len(hidden_dims) > 1:
            for i in range(1, len(hidden_dims)):
                # initialize axonal connections between previous layer and next layer
                layers.append(AxonLinear(hidden_dims[i-1], hidden_dims[i]))
                # initialize network layer
                layers.append(FDLayer(hidden_dims[i], ode_solver_steps))
        
        # wrap layers of facilitation-depression network in Sequential container
        self.fd_layers = nn.Sequential(*layers)

        # create parameters for output mapping, if any
        if output_mapping in ['affine', 'linear']:
            self.output_weight = nn.Parameter(
                torch.zeros((hidden_dims[-1],)))
        if output_mapping == 'affine':
            self.output_bias = nn.Parameter(
                torch.zeros((hidden_dims[-1],)))
        
        self.output_mapping = output_mapping

    # forward propagation of inputs through facilitation-depression network
    def forward(self, x):
        output = self.fd_layers(x)[...,-1,:]
        #output = self.fd_layers(x).sum(dim = -2)

        # output mapping; default is EPSC from last network layer
        if self.output_mapping in ['affine', 'linear']:
            output = output * self.output_weight
        if self.output_mapping == 'affine':
            output = output + self.output_bias

        return output


# input dimension is not specified
# input is assumed to be injected calcium current for each of hidden_dim neurons
class FDLayer(nn.Module):

    def __init__(self, hidden_dim, ode_solver_steps):
        super(FDLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_solver_steps

        # intracellular calcium concentration during rest state
        # constrain parameter to be positive
        self.log_Ca_mu = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # significance of current intracellular calcium concentration
        #   deviance from rest state
        # constrain parameter to be positive
        self.log_Ca_sigma = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # intracellular calcium concentration decay time constant
        # constrain parameter to be positive
        self.log_tau_Ca = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # parameter quantifying (K_Ca / tau_Ca)
        # constrain parameter to be positive
        self.log_alpha = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # EPSC decay time constant
        # constrain parameter to be positive
        self.log_tau_EPSC = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # parameter quantifying (n * N_total * K_Glu / tau_EPSC)
        # constrain parameter to be positive
        self.log_beta = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # maximum probability of vesicle release
        # constrain parameter to (0,1) interval
        self.presigmoid_P_rel_max = nn.Parameter(torch.zeros((hidden_dim,)))
        
        # minimum recovery rate from empty to releasable state
        # constrain parameter to be positive
        self.log_k_recov_min = nn.Parameter(torch.zeros((hidden_dim,)))

        # parameter quantifying (k_recov_max - k_recov_min)
        # constrain parameter to be positive
        self.log_k_recov_delta = nn.Parameter(torch.zeros((hidden_dim,)))

    # numerical ODE solver step
    def ODE_solver(self, I_Ca_t, EPSC, Ca, R_rel):
        dt = 1/self.ode_steps

        for _ in range(self.ode_steps):
            # current intracellular calcium concentration deviance from rest state
            Ca_diff = Ca - self.log_Ca_mu.exp()
            # sigmoid function component
            sigmoid = torch.sigmoid(Ca_diff / self.log_Ca_sigma.exp())
            # probability of vesicle release
            P_rel = torch.sigmoid(self.presigmoid_P_rel_max) * sigmoid
            # intermediate value
            temp = P_rel * R_rel * I_Ca_t
            # update excitatory postsynaptic current
            EPSC = EPSC - dt * (EPSC / self.log_tau_EPSC.exp() + self.log_beta.exp() * temp)
            # recovery rate from empty to releasable state
            k_recov = self.log_k_recov_min.exp() + self.log_k_recov_delta.exp() * sigmoid
            # update remaining ratio of vesicles releasable
            R_rel = R_rel + dt * (k_recov * (1 - R_rel) - temp)
            # update intracellular calcium concentration
            Ca = Ca + dt * (self.log_alpha.exp() * I_Ca_t - Ca_diff / self.log_tau_Ca.exp())
        
        # return hidden and cell states
        return EPSC, Ca, R_rel
    
    def forward(self, I_Ca):
        device = I_Ca.device

        # EPSC at initial rest state is zeros
        EPSC = torch.zeros((self.hidden_dim,), device = device)
        # Ca at initial rest state is Ca_mu
        Ca = self.log_Ca_mu.exp()
        # R_rel at initial rest state is ones
        R_rel = torch.ones((self.hidden_dim,), device = device)

        # propagate injected calcium current through layer nodes
        seq = []
        for t in range(I_Ca.size(-2)):
            # injected calcium current at time t
            I_Ca_t = I_Ca[...,t,:]
            # numerical ODE solver step
            EPSC, Ca, R_rel = self.ODE_solver(I_Ca_t, EPSC, Ca, R_rel)
            # save EPSC at time t
            seq.append(EPSC)
        
        # output is EPSC at each timestep
        output = torch.stack(seq, dim = -2)

        return output


# temporary solution to be replaced by future work
# axons here are interpreted as resistors in an electric circuit
# each neuron from previous layer has an axonal connection to each neuron in next layer
# EPSC from previous layer is directly passed as injected current for next layer
#   EPSC splits across axons according to their weights
#   weights for all axons from a given neuron sum to one 
class AxonLinear(nn.Module):
    
    def __init__(self, in_neurons, out_neurons):
        super(AxonLinear, self).__init__()
        # initial axon weights are equal and constrained positive
        # weights of important axonal connections will increase through learning
        self.log_weight = nn.Parameter(torch.zeros((in_neurons, out_neurons)))

    # propagate EPSC from previous layer to next layer
    def forward(self, EPSC):
        # EPSC has negative sign and must be converted to positive sign
        # enforce axon weights from a neuron sum to one
        return -1*EPSC @ torch.softmax(self.log_weight.exp(), dim = -1)
















