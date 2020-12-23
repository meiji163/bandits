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

MOVES = (0,1,2)
WIN = torch.Tensor([[ 0, -1,  1],
                    [ 1,  0, -1],
                    [-1,  1,  0]])

class ucbJanken():
    '''UCB algorithm with epsilon-greedy selection
    kwargs:
        gamma (float): exploration constant
        epsilon (float): probability of choosing randomly
        reset_prob (float): probability of resetting
    '''
    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma", 0.5)
        self.epsilon = kwargs.get("epsilon", 0.1)

        self.explore = Bernoulli(torch.tensor(self.epsilon))
        self.visits = [0,0,0]
        self.rewards = [0.,0.,0.]
    
    def __str__(self):
        return f"ucb: gamma = {self.gamma:.3f}, epsilon = {self.epsilon:.3f}"

    def observe(self, move, reward):
        m = move.item() if isinstance(move, torch.Tensor) else move
        r = reward.item() if isinstance(reward, torch.Tensor) else reward
        self.rewards[m] += r

    def ucb(self, m):
        if self.visits[m] == 0:
            return 0 
        return self.rewards[m]/self.visits[m]\
               + self.gamma*sqrt(sum(self.visits))/self.visits[m]

    def throw(self):
        if sum(self.visits) == 0:
            m = randint(0,2) 
        else:
            r = self.explore.sample()
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
        d = (1-self.epsilon)*d + (self.epsilon/3.0)*torch.ones(3)
        return d

ucb = ucbJanken(epsilon = 0.3, gamma = 3.0)
last_move = None

def ucb_agent( observation, configuration):
    global last_move
    global WIN
    if observation.step > 0:
        ucb.observe(last_move,
                        WIN[last_move, observation.lastOpponentAction])  

    last_move = ucb.throw().item()
    return last_move
