from math import sqrt, log
from random import randint
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

CHOKI = 0
GUU = 1
PAA = 2
MOVES = (CHOKI, GUU, PAA)

#WIN[i,j] = reward for playing i against j
WIN = torch.Tensor([[ 0, -1,  1],
                    [ 1,  0, -1],
                    [-1,  1,  0]])

def expected_reward(dist_1, dist_2):
    return dist_1 @ WIN @ dist_2

class JankenRNN(nn.Module):
    '''A GRU to play Janken. 
    Input: previous opponent move
    Output: policy distribution
    '''
    def __init__(self, device = None):
        super(JankenRNN, self).__init__()
        self.id = 0  
        self.device = device 
        self.hidden_size = 128 
        self.gru = nn.GRU(3, self.hidden_size,
                                num_layers = 3, 
                                dropout = 0.2,
                                batch_first = True)
        self.lin = nn.Linear(self.hidden_size, 3)
        self.softmax = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()

        self.hidden_state = None
        self.dist = 1/3*torch.ones(3)

    def __str__(self):
        return f"RNN {self.id}"

    def forward(self, inputs, h = None):
        x = F.one_hot(inputs, 3).float()
        if h is None:
            y, new_h = self.gru(x)
        else:
            y, new_h = self.gru(x, h)
        out = self.lin(self.relu(y[:,-1]))
        return self.softmax(out), new_h
    
    def observe(self, move, reward, device = None):
        last = ((move - reward)%3).long().unsqueeze(0)
        if device:
            last = last.to(device) 
        out, h = self.forward(last.unsqueeze(0), self.hidden_state)
        self.hidden_state = h.data
        self.dist = out.squeeze(0)
 
    def throw(self):
        policy = Categorical(self.dist)
        return policy.sample()
    
class dumbJanken():
    '''Dumb janken player chooses a random policy distribution and randomly resets it.
    kwargs:
        reset_prob (float): probability of resetting at each turn
        dists: list of distributions to use
        bias (float): scalar to determine biasing towards moves that previously won'''
    def __init__(self, reset_prob = 0.015, **kwargs):
        #expected reset time = 1/(reset_prob)
        self.reset = Bernoulli(torch.tensor([reset_prob]))
        self.dists = kwargs.get("dists")
        self.bias = kwargs.get("bias")

        if self.dists:
            self.policy = Categorical(dists.pop(0))
        else:
            self.policy = self.rand_dist()
        
    def __str__(self):
        return f"dumb: reset_prob = {self.reset.probs.item()}, bias = {self.bias}"
    def rand_dist(self):
        return Categorical(torch.rand(3))

    def throw(self):
        r = self.reset.sample().item()
        if r == 1:
            if self.dists:
                self.policy = Categorical(dists.pop(0))
            else:
                self.policy = self.rand_dist()
        return self.policy.sample() 

    def observe(self, move, reward):
        if reward == 1 and self.bias:
            v = F.one_hot(move, 3)
            self.policy.probs = self.bias*v + (1-self.bias)*self.policy.probs

    @property
    def dist(self):
        return self.policy.probs
        
class exp3rJanken():
    '''Implements EXP3 with Resets 
    (`The non-stationary stochastic multi-armed bandit problem` 2017)
    kwargs
        gamma (float): exploration constant between 0 and 1
        H (int): observation constant
        delta (float): probability of failure in drift detection
    '''
    def __init__(self, **kwargs):
        self.H = kwargs.get("H",360)
        self.gamma = kwargs.get("gamma", 0.15)
        self.epsilon = sqrt( (-3*log(0.1))/(2*self.gamma*self.H) )

        self.rewards = [0.,0.,0.]
        self.observations = [0,0,0]
        self.weights = torch.ones(3)
        self.policy = Categorical(self.weights)

    def __str__(self):
        return f"exp3r: gamma = {self.gamma}, epsilon = {self.epsilon}"
    def observe(self, move, reward):
        reward = (reward + 1)/2.0
        self.rewards[move] += reward
        self.observations[move] += 1
        self.weights[move] *= torch.exp( self.gamma*reward/(3*self.policy.probs[move]) )
        probs = ((1 - self.gamma)/self.weights.sum())*self.weights\
                + (self.gamma/3.0)*torch.ones(3) 
        self.policy = Categorical(probs)

        #reset if drift is detected 
        if min(self.observations) > self.gamma*self.H/3.0:
            means = [self.rewards[i]/self.observations[i] for i in range(3)]
            prev_means = copy(means)
            prev_means[move] = (self.rewards[move] - reward)/(self.observations[move] -1)
            prev_max = max(prev_means)

            if any([means[i] - prev_max >= self.epsilon for i in range(3)]):
                self.reset()
            
    def reset(self):
        self.weights = torch.ones(3)
        self.policy = Categorical(self.weights)
        self.rewards = [0,0,0]
        self.observations = [0,0,0]    

    def throw(self):
        return self.policy.sample()

    @property
    def dist(self):
        return self.policy.probs

class ucbJanken():
    '''UCB algorithm with epsilon-greedy selection
    kwargs:
        gamma (float): exploration constant
        epsilon (float): probability of choosing randomly'''
    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma", 0.5)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.coin = Bernoulli(torch.tensor(self.epsilon))
        self.visits = [0,0,0]
        self.rewards = [0.,0.,0.]
    
    def __str__(self):
        return f"ucb: gamma = {self.gamma}, epsilon = {self.epsilon}"
    def observe(self, move, reward):
        '''move (torch.tensor)'''
        self.rewards[move.item()] += reward

    def ucb(self, m):
        if self.visits[m] == 0:
            return 0 
        return self.rewards[m]/self.visits[m]\
               + self.gamma*sqrt(log( sum(self.visits) ))/self.visits[m]

    def throw(self):
        if sum(self.visits) == 0:
            m = randint(0,2) 
        else:
            r = self.coin.sample()
            if r.item() == 1:
                m = randint(0,2)
            else:
                m = max( MOVES, key = self.ucb) 
        self.visits[m] += 1
        return torch.tensor(m)
    
    @property
    def dist(self):
        if sum(self.visits) == 0:
            return 1/3*torch.ones(3)
        best = max(MOVES, key = self.ucb)
        d = torch.zeros(3)
        d[best] = 1.0
        d = (1-self.epsilon)*d + (self.epsilon/3)*torch.ones(3)

class pucbJanken(ucbJanken):
    '''PUCB algorithm with RNN predictor
    args:
        predictor (JankenRNN)'''
    def __init__(self, predictor, **kwargs):
        super(pucbJanken, self).__init__(**kwargs)
        self.predictor = predictor

    def observe(self, move, reward):
        self.predictor.observe(move)
        self.rewards[move.item()] += reward

    def pucb(self, m):
        if self.visits[m] == 0:
            return 0 
        prior = self.predictor.dist[m].item()
        return self.rewards[m]/self.visits[m]\
               + self.gamma*prior*sqrt(log( sum(self.visits) ))/self.visits[m]

    def throw(self):
        if sum(self.visits) == 0:
            m = randint(0,2) 
        else:
            r = self.coin.sample()
            if r.item() == 1:
                m = randint(0,2)
            else:
                m = max(MOVES, key = self.pucb) 
        self.visits[m] += 1
        return torch.tensor(m)
