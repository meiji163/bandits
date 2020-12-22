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

class serJanken():
    def __init__(self, **kwargs):
        self.delta = kwargs.get("delta", 0.35)
        self.epsilon = kwargs.get("epsilon", 0.3)
        reset_prob = kwargs.get("reset_prob", 0.05)
        self.coin = Bernoulli(torch.tensor( reset_prob))
        self.means = [0.,0.,0.] 
        self.arms = {0,1,2}
        self.not_played = [0,1,2]

        self.thresh = int(log(3.0/self.delta))
        self.round = 1
        self.best = None

    def throw(self):
        if self.best is not None:
            return self.best
        
        k = randint(0, len(self.not_played)-1)
        m = self.not_played.pop(k) 
        if not self.not_played:
            self.round += 1
            self.not_played = list(self.arms) 
        return m

    def observe(self, move, reward):
        m = move.item() if isinstance(move, torch.Tensor) else move
        r = reward.item() if isinstance(reward, torch.Tensor) else reward

        norm_r = (r + 1)/2
        self.means[m] = (self.round - 1)/(self.round)* self.means[m] \
                        + norm_r/self.round

        if self.best is not None:
            if max(self.means) - self.means[self.best] > self.epsilon:
                self.reset() 
            return 

        flip = self.coin.sample()
        if flip.item() == 1:
            self.reset()
        #elimination
        if self.round >= self.thresh:
            max_mean = max( self.arms, key = lambda i: self.means[i])
            elim = set()
            for m in self.arms:
                if max_mean - self.means[m] + self.epsilon \
                    >= sqrt(1/(2*self.round) * log( 12*self.round**2/self.delta)):
                    elim.add(m) 
            self.arms -= elim 
            if len(self.arms) == 1:
                self.best = self.arms.pop()

    def reset(self):
        self.round = 1
        self.arms = {0,1,2}
        self.not_played = [0,1,2]
        self.means = [0.,0.,0.]
        self.best = None

sj = serJanken()
last_move = None
print(sj.thresh, sj.epsilon)
def ser_agent( observation, configuration):
    global last_move
    global WIN
    if observation.step > 0:
        sj.observe(last_move,
                        WIN[last_move, observation.lastOpponentAction])  

    last_move = sj.throw() 
    return last_move

class Observation():
    def __init__(self, move = None, step = 0):
        self.lastOpponentAction = move
        self.step = step

if __name__ == "__main__":
    obs = Observation()
    score = 0
    while True:
        m = ser_agent(obs, None)
        print(f"Score:\t {score}")
        obs.lastOpponentAction = int( input("Your Move:\t"))
        assert 0 <= obs.lastOpponentAction < 3
        obs.step += 1
        result = WIN[ obs.lastOpponentAction, m].item()
        score += result
