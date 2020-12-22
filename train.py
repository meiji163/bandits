#!/usr/bin/env python3
import argparse
from janken import *
from random import randint
from glob import glob
import os
import torch

def train(bot, bot_op, optimizer, **kwargs):
    interval = kwargs.get("interval", 30)
    n_it = kwargs.get("n_it", 50)
    device = kwargs.get("device")    
    
    rewards = []
    exp_rewards = []
    for _ in range(n_it):
        dists = []
        op_dists = []
        for i in range(interval):
            dists.append(bot.dist)
            op_dists.append(bot_op.dist)

            m1 = bot.throw()
            m2 = bot_op.throw()
            reward = WIN[m1,m2]
            rewards.append(reward.item())

            m1 = m1.to(device)
            bot.observe(m1, reward)
            bot_op.observe(m2, -reward)
        
        exp_reward = 0.
        for i in range(len(dists)):
            exp_reward += expected_reward( dists[i], op_dists[i], device)
        exp_reward /= interval
        loss = -exp_reward
        exp_rewards.append(exp_reward.item())

        #backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return exp_rewards, rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "training script for Janken bot")
    parser.add_argument("-n", help = "number of opponents per epoch", type = int, dest = 'n', default = 10)
    parser.add_argument("-g", help = "number of games per opponent", type = int, dest = 'g', default = 10)
    parser.add_argument("-e", help = "number of epochs", type = int, dest = 'e', required = True)
    parser.add_argument("-f", help = "file to write stats", type = str, dest = 'f',\
                              default = os.path.join( os.getcwd(), "stats.txt"))
    args = parser.parse_args()    

    device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available else "cpu")
 
    j = JankenRNN(device)
    j.train()
    j.to(device)
    weight_paths = glob("weights/j_*.pt")
    if len(weight_paths) == 0:
        torch.save( {"model_state_dict": j.state_dict(), "id": 0}, os.path.join("weights", "j_0.pt"))
    else:
        idx = max(range( len(weight_paths)), key = lambda x: int(weight_paths[x][-4]))
        weight = torch.load( weight_paths[idx], map_location = device)
        print(f"Loading {weight_paths[idx]}")
        j.load_state_dict(weight["model_state_dict"])
        j.id = weight["id"]

    optimizer = torch.optim.AdamW(j.parameters(), lr = 1e-3)
    optim_path = os.path.join( os.getcwd(), "weights", f"adam_w.pt")
    if os.path.exists(optim_path):
        param = torch.load(optim_path, map_location = device)
        optimizer.load_state_dict(param)
        #move optimizer states to GPU
        if device.type == "cuda":
            for state in optimizer.state.values():
                for k, t in state.items():
                    if torch.is_tensor(t):
                        state[k] = t.cuda()

    games = args.g #number of games to play against each opponent 
    for epoch in range(args.e):
        for _ in range(args.n):
            #choose the opponent and random hyperparameters
            opps = ["dumb", "exp3r", "ucb"] + 2*["rnn"]
            r = randint(0, len(opps)-1)

            if opps[r] == "dumb":
                reset_time = randint(2,30)
                b = 0.2*torch.rand(1).item()
                j_op = dumbJanken(bias = b, reset_prob = 1/reset_time)

            elif opps[r] == "exp3r":
                exploration = 0.1 + 0.2*torch.rand(1)
                H = randint(100,300)
                j_op = exp3rJanken(gamma = exploration.item(), H = H)

            elif opps[r] == "ucb":
                exploration = 5*torch.rand(1)
                e = 0.5*torch.rand(1)
                j_op = ucbJanken(gamma = exploration.item(),
                                 epsilon = e.item())

            elif opps[r] == "rnn" or opps[r] == "pucb":
                n = randint(0, len(weight_paths) -1)
                weights = torch.load(weight_paths[n], map_location = device)
                p = JankenRNN()
                p.load_state_dict(weights["model_state_dict"])
                p.id = weights["id"]
                p.eval()
                if opps[r] == "pucb":
                    j_op = pucbJanken(p)
                else:
                    j_op = p
            
            print(f"\nEpoch {epoch+1} --- playing vs {j_op}")
            stats = []            
            for _ in range(games):
                avg, scores = train(j, j_op, optimizer, device = device)
                print(f"expected: {100*sum(avg)/len(avg):.2f}\t\t score: {100*sum(scores)/len(scores):.2f}")
                stats.extend(avg)
            
            with open(args.f, 'a') as f:
                f.write(f"Epoch {epoch + 1} --- {j_op}\n")
                f.write(",".join([format(x, '.3f') for x in stats]) + '\n')  

        j.id += 1
        out_path = os.path.join( os.getcwd(), "weights", f"j_{j.id}.pt")
        weight_paths.append(out_path)
        torch.save( {"model_state_dict": j.state_dict(), "id": j.id}, out_path)

    torch.save( optimizer.state_dict(), optim_path)
