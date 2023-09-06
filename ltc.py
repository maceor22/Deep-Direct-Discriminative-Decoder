from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import autograd as ag
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import time as time



class LTC(nn.Module):
    
    def __init__(
            self, input_dim, hidden_dim, use_cell_memory = False, 
            ode_solver_unfolds = 6, epsilon = 1e-8,
            input_mapping = 'affine', output_mapping = 'affine',
            ):
        super(LTC, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        if use_cell_memory:
            self.lstm_cell = LSTMCell(input_dim, hidden_dim)
        self.cell_memory = use_cell_memory
        
        self.ltc_cell = LTCCell(
            input_dim = input_dim, hidden_dim = hidden_dim,
            ode_solver_unfolds = ode_solver_unfolds, epsilon = epsilon,
            input_mapping = input_mapping, output_mapping = output_mapping,
            )
        
    def forward(self, x, state = None):
        device = x.device
        
        if state is None:
            hidden_state = torch.zeros((x.size(0), self.hidden_dim), device = device)
            if self.cell_memory:
                cell_state = torch.zeros((x.size(0), self.hidden_dim), device = device)
        else:
            if self.cell_memory:
                hidden_state, cell_state = state
            else:
                hidden_state = state
        
        seq = []
        for t in range(x.size(1)):
            
            xt = x[:,t,:]
            
            if self.cell_memory:
                hidden_state, cell_state = self.lstm_cell(
                    xt, (hidden_state, cell_state))
            
            hidden_output, hidden_state = self.ltc_cell(xt, hidden_state)
            seq.append(hidden_output.unsqueeze(1))
        
        output = torch.cat(seq, dim = 1)
        state = (hidden_state, cell_state) if self.cell_memory else hidden_state
        
        return output, state


class LTCCell(nn.Module):
    
    def __init__(
            self, input_dim, hidden_dim, ode_solver_unfolds = 6,
            epsilon = 1e-8,
            input_mapping = 'affine', output_mapping = 'affine',
            ):
        super(LTCCell, self).__init__()
        
        self.ode_unfolds = ode_solver_unfolds
        self.epsilon = epsilon
        
        if input_mapping == 'affine' or input_mapping == 'linear':
            self.input_weight = nn.Parameter(torch.ones((input_dim,)))
        if input_mapping == 'affine':
            self.input_bias = nn.Parameter(torch.zeros((input_dim,)))            
        self.input_mapping = input_mapping
        
        
        self.sensing_mu = nn.Parameter(
            torch.zeros((input_dim, hidden_dim)).uniform_(0.3, 0.8))
        
        self.sensing_sigma = nn.Parameter(
            torch.zeros((input_dim, hidden_dim)).uniform_(3, 8))
        
        self.sensing_w = nn.Parameter(
            torch.zeros((input_dim, hidden_dim)).uniform_(0.01, 1))
        
        self.sensing_erev = nn.Parameter(
            2*torch.randint(0, 2, size = (input_dim, hidden_dim)).float()-1)
        
        
        self.internal_mu = nn.Parameter(
            torch.zeros((hidden_dim, hidden_dim)).uniform_(0.3, 0.8))
        
        self.internal_sigma = nn.Parameter(
            torch.zeros((hidden_dim, hidden_dim)).uniform_(3, 8))
        
        self.internal_w = nn.Parameter(
            torch.zeros((hidden_dim, hidden_dim)).uniform_(0.01, 1))
        
        self.internal_erev = nn.Parameter(
            2*torch.randint(0, 2, size = (hidden_dim, hidden_dim)).float()-1)
        
        
        self.vleak = nn.Parameter(
            torch.zeros((hidden_dim,)).uniform_(-0.2, 0.2))
        
        self.gleak = nn.Parameter(torch.ones((hidden_dim,)))
        
        self.cm = nn.Parameter(torch.ones((hidden_dim,))/2)
        
        
        if output_mapping == 'affine' or output_mapping == 'linear':
            self.output_weight = nn.Parameter(torch.ones((hidden_dim,)))
        if output_mapping == 'affine':
            self.output_bias = nn.Parameter(torch.zeros((hidden_dim,)))            
        self.output_mapping = output_mapping
    
    def ODE_solver(self, x, h0, time_elapsed):
        V = h0
        
        sensing_w_activated = self.sensing_w * self.sigmoid(
            x, self.sensing_mu, self.sensing_sigma)
        
        sensing_rev_activated = sensing_w_activated * self.sensing_erev
        
        w_numerator_sensing = sensing_rev_activated.sum(dim = 1)
        w_denominator_sensing = sensing_w_activated.sum(dim = 1)
        
        cm_t = self.cm / (time_elapsed / self.ode_unfolds)
        
        for t in range(self.ode_unfolds):
            w_activated = self.internal_w * self.sigmoid(
                V, self.internal_mu, self.internal_sigma)
            
            rev_activated = w_activated * self.internal_erev
            
            w_numerator = rev_activated.sum(dim = 1) + w_numerator_sensing
            w_denominator = w_activated.sum(dim = 1) + w_denominator_sensing
            
            numerator = V * cm_t + self.gleak * self.vleak + w_numerator
            denominator = cm_t + self.gleak + w_denominator
            
            V = numerator / (denominator + self.epsilon)
        
        return V
        
    def sigmoid(self, V, mu, sigma):
        return torch.sigmoid(sigma * (V.unsqueeze(-1) - mu))
        
    def input_mapper(self, inputs):
        if self.input_mapping == 'affine' or self.input_mapping == 'linear':
            inputs = inputs * self.input_weight
        if self.input_mapping == 'affine':
            inputs = inputs + self.input_bias
        return inputs
            
    def output_mapper(self, outputs):
        if self.output_mapping == 'affine' or self.output_mapping == 'linear':
            outputs = outputs * self.output_weight
        if self.output_mapping == 'affine':
            outputs = outputs + self.output_bias
        return outputs
    
    def forward(self, x, h0, time_elapsed = 1.0):
        x = self.input_mapper(x)
        
        hidden_state = self.ODE_solver(x, h0, time_elapsed)
        
        output = self.output_mapper(hidden_state)
        
        return output, hidden_state
        

class LSTMCell(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        
        self.input_mapper = nn.Linear(input_dim, hidden_dim*4)
        self.hidden_mapper = nn.Linear(hidden_dim, hidden_dim*4)
        
    def forward(self, x, states):
        h0, c0 = states
        
        input_gate, forget_gate, cell_gate, output_gate = torch.chunk(
            self.input_mapper(x) + self.hidden_mapper(h0), 
            chunks = 4, dim = 1)
        
        i = torch.sigmoid(input_gate)
        f = torch.sigmoid(forget_gate)
        g = torch.tanh(cell_gate)
        o = torch.sigmoid(output_gate)
        
        cell_state = f * c0 + i * g
        hidden_state = o * torch.tanh(cell_state)
        
        return hidden_state, cell_state
















