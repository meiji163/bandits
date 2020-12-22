from math import sqrt, log, exp
from random import randint
from copy import copy
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli

# Janken convention:
# CHOKI = 0, GUU = 1, PAA = 2 

# Kaggle convention:
# GUU = 0, PAA = 1, CHOKI = 2

WIN = torch.Tensor([[ 0, -1,  1],
                    [ 1,  0, -1],
                    [-1,  1,  0]])

class exp3rJanken():
    '''Implements EXP3 with Resets 
    (`The non-stationary stochastic multi-armed bandit problem` 2017)
    kwargs
        gamma (float): exploration constant between 0 and 1
        H (int): observation constant
        delta (float): probability of failure in drift detection
    '''
    def __init__(self, **kwargs):
        self.H = kwargs.get("H",200)
        self.gamma = kwargs.get("gamma", 0.1)
        self.epsilon = 0.5*sqrt( (-3*log(0.1))/(2*self.gamma*self.H))

        self.rewards = [0.,0.,0.]
        self.observations = [0,0,0]
        self.weights = torch.ones(3)
        self.policy = Categorical(self.weights)

    def observe(self, move, reward):
        m = move.item() if isinstance(move, torch.Tensor) else move
        r = reward.item() if isinstance(reward, torch.Tensor) else reward
        norm_r = (reward + 1)/2.0
        self.rewards[m] += norm_r
        self.observations[m] += 1
        prior = self.dist[m].item() 
        if self.weights[m] < 1e15:
            self.weights[m] *= exp( self.gamma*norm_r/(10*prior) )
            probs = ((1 - self.gamma)/self.weights.sum())*self.weights\
                    + (self.gamma/3.0)*torch.ones(3) 
            self.policy = Categorical(probs)

        #reset if drift is detected 
        if min(self.observations) > self.gamma*self.H/3.0:
            means = [self.rewards[i]/self.observations[i] for i in range(3)]
            prev_means = copy(means)
            prev_means[move] = (self.rewards[move] - norm_r.item())/(self.observations[move] -1)
            prev_max = max(prev_means)

            if any([means[i] - prev_max >= self.epsilon for i in range(3)]):
                self.reset()
            
    def reset(self):
        self.weights = torch.ones(3)
        self.policy = Categorical(self.weights)
        self.rewards = [0.,0.,0.]
        self.observations = [0,0,0]    

    def throw(self):
        return self.policy.sample()

    @property
    def dist(self):
        return self.policy.probs

expj = exp3rJanken(gamma = 0.15, H = 250)
last_move = None

def exp3r_agent( observation, configuration):
    global last_move
    global WIN
    if observation.step > 0:
        expj.observe(last_move,
                        WIN[last_move, observation.lastOpponentAction])  

    last_move = expj.throw().item()
    return last_move
