from math import sqrt, log, exp
from random import randint, choice
from copy import copy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta

CHOKI = 0
GUU = 1
PAA = 2
MOVES = (CHOKI, GUU, PAA)

#WIN[i,j] = reward for playing i against j
WIN = torch.Tensor([[ 0, -1,  1],
                    [ 1,  0, -1],
                    [-1,  1,  0]])

UNIFORM = 1/3*torch.ones(3)

def expected_reward(dist_1, dist_2, device = None):
    global WIN 
    if device is not None:
        return dist_1.to(device)@ WIN.to(device)@ dist_2.to(device)
    return dist_1 @ WIN @ dist_2

def counter_policy(dist, epislon = 0.25, device = None):
    global WIN
    global BASIS
    global UNIFORM
    if device is not None:
        WIN = WIN.to(device)
        BASIS = BASIS.to(device)
        UNIFORM = UNIFORM.to(device)

    q = WIN @ dist
    i = (torch.argmax(q) - 1)%3    
    n = WIN[i]
    norms = 1/6*WIN @ WIN
    counter = (q @ n)/(n @ n)*n + norms[i] + UNIFORM
    return epislon*UNIFORM + (1-epislon)*counter

def max_reward(dist, device = None):
    return expected_reward(counter_policy(dist), dist, device)

class rnnJanken(nn.Module):
    def __init__(self, model_type = "GRU"):
        super(rnnJanken, self).__init__()
    
        self.id = 0
        self.hidden_size = 32
        self.model_type = model_type
        if model_type == "GRU":
            self.rnn = nn.GRU(9, self.hidden_size,
                                    num_layers = 4, 
                                    dropout = 0.1,
                                    bias = False,
                                    batch_first = True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(9, self.hidden_size,
                                    num_layers = 3,
                                    bias = False,
                                    batch_first = True,
                                    dropout = 0.1)
        else:
            raise ValueError("model_type must be `GRU` or `LSTM`")

        self.lin = nn.Linear(self.hidden_size, 3)
        self.hidden_state = None
        self.softmax = nn.Softmax(dim = 1)
        self.dist = UNIFORM
 
    def __str__(self):
        return f"{self.model_type}: {self.id}"
     
    def forward(self, inputs, state = None):
        if state is None:
            y, new_state = self.rnn(inputs)
        else:
            y, new_state = self.rnn(inputs)
        out = self.lin(y[:,-1])
        return out, new_state 

    def throw(self):
        policy = Categorical(self.dist)
        return policy.sample()

    def observe(self, move, reward, device = None):
        opp_move = ((move - reward)%3).unsqueeze(0)
        move = move.unsqueeze(0)
        inputs = self.encode(opp_move, move, device)
    
        out, h = self.forward(inputs, self.hidden_state)
        self.hidden_state = h
        self.dist = counter_policy(self.softmax(out).squeeze(0), device)
    
    def encode(self, opp_moves, moves, device = None):
        if device is not None:
            moves = moves.to(device)
            opp_moves = opp_moves.to(device)
        x = opp_moves + 3*moves
        encoded = F.one_hot(x.long(), 9).float()
        if encoded.dim() == 2:
            return encoded.unsqueeze(0)
        return encoded

    def reset(self):
        self.hidden_state = None
        self.dist = UNIFORM 

class randJanken():
    '''Random janken player chooses a random policy distribution and randomly resets it.
    kwargs:
        reset_prob (float): probability of resetting at each turn
        dists: list of distributions to use
        bias (float): scalar to determine biasing towards moves that previously won'''
    def __init__(self, reset_prob = 0.015, **kwargs):
        #expected reset time = 1/(reset_prob)
        if reset_prob == 0:
            self.coin = None
        else:
            self.coin = Bernoulli(torch.tensor(reset_prob))
        self.dists = kwargs.get("dists")
        self.bias = kwargs.get("bias")

        if self.dists:
            self.policy = Categorical(dists.pop(0))
        else:
            self.policy = self.rand_dist()
        
    def __str__(self):
        if self.coin is None:
            return "uniform"
        return f"rand: reset_prob = {self.coin.probs.item():.3f}, bias = {self.bias:.3f}"
    def rand_dist(self):
        return Categorical(torch.rand(3))

    def throw(self):
        r = 0
        if self.coin is not None:
            r = self.coin.sample().item()
        if r == 1:
            if self.dists:
                self.policy = Categorical(dists.pop(0))
            else:
                self.reset()
        return self.policy.sample() 

    def observe(self, move, reward):
        if reward == 1 and self.bias:
            v = F.one_hot(move, 3)
            self.policy.probs = self.bias*v + (1-self.bias)*self.policy.probs
    
    def reset(self):
        self.policy = self.rand_dist()

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
        self.H = kwargs.get("H",200)
        self.gamma = kwargs.get("gamma", 0.1)
        self.epsilon = 0.5*sqrt( (-3*log(0.1))/(2*self.gamma*self.H))

        self.rewards = [0.,0.,0.]
        self.observations = [0,0,0]
        self.weights = torch.ones(3)
        self.policy = Categorical(self.weights)

    def __str__(self):
        return f"exp3r: gamma = {self.gamma:.3f}, epsilon = {self.epsilon:.3f}"

    def observe(self, move, reward):
        norm_r = (reward + 1)/2.0
        m = move.item()
        self.rewards[m] += norm_r.item()
        self.observations[m] += 1
        prior = self.dist[m].item() 
        if self.weights[m] < 1e15:
            self.weights[m] *= exp( self.gamma*norm_r.item()/(100*prior) )
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
        self.reset_prob = kwargs.get("reset_prob", 0.2)
        self.coin = Bernoulli(torch.tensor(self.reset_prob))

        self.explore = Bernoulli(torch.tensor(self.epsilon))
        self.visits = [0,0,0]
        self.rewards = [0.,0.,0.]
    
    def __str__(self):
        return f"ucb: gamma = {self.gamma:.3f}, epsilon = {self.epsilon:.3f}"

    def observe(self, move, reward):
        m = move.item() if isinstance(move, torch.Tensor) else move
        r = reward.item() if isinstance(reward, torch.Tensor) else reward
        flip = self.coin.sample()
        if flip.item() == 1:
            self.reset()
        self.rewards[move.item()] += reward.item()

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
    
    def reset(self):
        self.visits = [0,0,0]
        self.rewards = [0.,0.,0.]        

    @property
    def dist(self):
        if sum(self.visits) == 0:
            return 1/3*torch.ones(3)
        best = max(MOVES, key = self.ucb)
        d = torch.zeros(3)
        d[best] = 1.0
        d = (1-self.epsilon)*d + (self.epsilon/3.0)*torch.ones(3)
        return d

class pucbJanken(ucbJanken):
    '''PUCB algorithm with RNN predictor
    args:
        predictor (gruJanken or lstmJanken)'''
    def __init__(self, predictor, **kwargs):
        super(pucbJanken, self).__init__(**kwargs)
        self.predictor = predictor

    def __str__(self):
        return f"pucb: {self.predictor.id}, gamma = {self.gamma:.3f}, reset_prob = {self.reset_prob:.3f}"

    def observe(self, move, reward, device = None):
        m = move.item() if isinstance(move, torch.Tensor) else move
        r = reward.item() if isinstance(reward, torch.Tensor) else reward
        flip = self.coin.sample()
        if flip.item() == 1:
            self.reset()

        self.predictor.observe(move, reward, device)
        self.rewards[move.item()] += reward.item()

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
            r = self.explore.sample()
            if r.item() == 1:
                m = randint(0,2)
            else:
                m = max(MOVES, key = self.pucb) 
        self.visits[m] += 1
        return torch.tensor(m)

    @property
    def dist(self):
        if sum(self.visits) == 0:
            return 1/3*torch.ones(3)
        best = max(MOVES, key = self.pucb)
        d = torch.zeros(3)
        d[best] = 1.0
        d = (1-self.epsilon)*d + (self.epsilon/3)*torch.ones(3)

class serJanken():
    def __init__(self, **kwargs):
        self.delta = kwargs.get("delta", 0.35)
        self.epsilon = kwargs.get("epsilon", 0.3)
        self.reset_prob = kwargs.get("reset_prob", 0.1)
        self.coin= Bernoulli(torch.tensor( self.reset_prob))
        self.means = [0.,0.,0.] 
        self.arms = {0,1,2}
        self.not_played = [0,1,2]

        self.thresh = int(log(3.0/self.delta))
        self.round = 1
        self.best = None

    def __str__(self):
        return f"ser4: thresh = {self.thresh}, reset_prob = {self.reset_prob:.3f}, epsilon = {self.epsilon:.3f}"

    def throw(self):
        if self.best is not None:
            return torch.tensor(self.best)
        k = randint(0, len(self.not_played) - 1) 
        m = self.not_played.pop(k) 
        if not self.not_played:
            self.round += 1
            self.not_played = list(self.arms) 
        return torch.tensor(m)

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
            return
        #elimination
        if self.round >= self.thresh:
            if len(self.arms) == 0:
                self.reset()
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
    
    @property
    def dist(self):
        if self.best is None:
            return UNIFORM
        else:
            d = torch.zeros(3)
            d[self.best] = 1
            return d

class copyJanken():
    def __init__(self, epsilon = 0.5):
        self.epsilon = epsilon
        self.explore = Bernoulli(torch.tensor(self.epsilon))
        self.last = None

    def __str__(self):
        return f"copy: epsilon = {self.epsilon:.3f}"

    def reset(self):
        pass
    
    def throw(self):
        if self.last is None:
            return torch.tensor(randint(0,2)) 
        else:
            return self.last

    def observe(self, move, reward):
        r = self.explore.sample()
        if r.item() == 1:
            return torch.tensor(randint(0,2))

    @property
    def dist(self):
        d = torch.zeros(3)
        d[self.last] = 1
        return d

class constJanken():
    def __init__(self, reset_prob = 0.5):
        if isinstance(reset_prob, torch.Tensor):
            self.coin = Bernoulli(reset_prob)
            self.reset_prob = reset_prob.item()
        else:
            self.reset_prob = reset_prob
            self.coin = Bernoulli(torch.tensor(reset_prob)) 
        self.move = randint(0,2)

    def __str__(self):
        return f"const: reset_prob = {self.reset_prob}"
    
    def throw(self):
        return torch.tensor(self.move)

    def observe(self,move, reward):
        r = self.coin.sample()
        if r.item() == 1:
            self.reset()

    def reset(self):
        self.move = randint(0,2)

class bayesJanken():
    def __init__(self, **kwargs):
        self.reset_prob = kwargs.get("reset_prob", 0.05)
        self.coin = Bernoulli( torch.tensor(self.reset_prob)) 
        self.rewards = Beta(torch.ones(3), torch.ones(3))
        
    def observe(self, move, reward):
        flip = self.coin.sample()
        if flip.item() == 1:
            self.reset()

        if reward == 1:
            self.rewards.concentration1[move] += reward
        else:
            self.rewards.concentration2[move] -= reward
    
    def __str__(self):
        return f"bayes: reset_prob = {self.reset_prob}"

    def reset(self):
        self.rewards = Beta( torch.ones(3), torch.ones(3))
       
    def throw(self):
        dist = self.rewards.sample()
        return torch.argmax(dist) 

class Observation():
    def __init__(self, move = None, step = 0):
        self.lastOpponentAction = move
        self.step = step

if __name__ == "__main__":
    j = lstmJanken()
    j.eval()
    w = torch.load("weights/j_32.pt", map_location = torch.device("cpu"))
    j.load_state_dict(w["model_state_dict"])
    score = 0
    while(True):
        print(f"Score:\t{score}")
        i = int(input("Your Move:\t"))
        m = j.throw()
        print(m.item())
        r = WIN[m,i]
        j.observe(m, r)
        print(j.dist)
        score -= r.item()
         
