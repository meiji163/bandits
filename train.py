#!usr/bin/env python3
import torch
import numpy as np

CHOKI = 0
GUU = 1
PAA = 2

WIN = np.matrix('0 -1  1 ;\
                 1  0 -1;\
                -1  1  0')   

def play_match(policy1, policy2, length = 1000):
    bot1 = JankenBot(policy1)
    bot2 = JankenBot(policy2)
    rewards = []
    loss = 0.0
    for i in range(length):
        a, dist = bot1.throw(dist = True)
        b = bot2.throw()

        reward = WIN[ a.item(), b.item()]
        rewards.append(reward)
        loss += - dist.log_prob(a)*reward

        bot1.update(mv1, mv2)
        bot2.update(mv2, mv1)
    loss /= length
    return loss

if __name__ == "__main__":
    
