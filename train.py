#!/usr/bin/env python3
import argparse
from janken import *
from random import randint
from glob import glob
import os
import torch

def train(bot, bot_op, optimizer, err, **kwargs):
    interval = kwargs.get("interval", 100)
    n_it = kwargs.get("n_it", 10)
    device = kwargs.get("device")    
    n_games = kwargs.get("n_games", 5)
    stats = kwargs.get("stats")
    
    for game in range(n_games):
        rewards = []
        hidden_states = []
        targets = []
        moves = []
        opp_moves = []
        for _ in range(n_it):
            for i in range(interval):
                m1 = bot.throw()
                m2 = bot_op.throw()
                
                reward = WIN[m1,m2]
                rewards.append(reward.item())

                m1 = m1.to(device)
                bot.observe(m1, reward)
                bot_op.observe(m2, -reward)
                
                if i == interval -1:
                    moves.append( m1.item())
                    opp_moves.append( m2.item())

            hidden_states.append( bot.hidden_state)
            targ = bot_op.throw().item()
            targets.append(targ)
        
        inputs = (torch.tensor(opp_moves).to(device), torch.tensor(moves).to(device))
        targets = torch.tensor(targets).to(device)
        hidden_states = torch.cat( hidden_states, dim = 1).to(device)
        outputs, _ = bot(*inputs, hidden_states)
        loss = err(outputs, targets)

        avg_reward = 0.5*(sum(rewards)/len(rewards) +1)
        loss *= (1- avg_reward) 
        #backpropagate
        optimizer.zero_grad()
        loss.backward(retain_graph = False if game == n_games -1 else True)
        optimizer.step()
        if stats is not None:
            stats.append(avg_reward)

        print(f"score: {sum(rewards):.0f}/{len(rewards)}\t loss: {loss.item()}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "training script for Janken bot")
    parser.add_argument("-n", help = "number of opponents per epoch", type = int, dest = 'n', default = 10)
    parser.add_argument("-g", help = "number of games per opponent", type = int, dest = 'g', default = 10)
    parser.add_argument("-e", help = "number of epochs", type = int, dest = 'e', required = True)
    parser.add_argument("-f", help = "file to write stats", type = str, dest = 'f',\
                              default = os.path.join( os.getcwd(), "stats.txt"))
    args = parser.parse_args()    

    device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available else "cpu")
 
    j = gruJanken(device)
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
    
    err = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(j.parameters(), lr = 0.1)
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

    for epoch in range(args.e):
        for _ in range(args.n):
            #choose the opponent and random hyperparameters
            opps = ["rand", "copy", "exp3r", "ucb", "unif"] + 4*["rnn"]
            r = randint(0, len(opps)-1)

            if opps[r] == "rand":
                reset_time = randint(2,100)
                b = 0.8*torch.rand(1).item()
                j_op = randJanken(bias = b, reset_prob = 1/reset_time)

            elif opps[r] == "unif":
                j_op = randJanken(bias = 0, reset_prob = 0)

            elif opps[r] == "exp3r":
                exploration = 0.1 + 0.2*torch.rand(1)
                H = randint(100,300)
                j_op = exp3rJanken(gamma = exploration.item(), H = H)

            elif opps[r] == "ucb":
                reset_time = randint(2,30)
                exploration = 5*torch.rand(1)
                e = 0.5*torch.rand(1)
                j_op = ucbJanken(gamma = exploration.item(),
                                 epsilon = e.item(),
                                 reset_prob = 1/reset_time)

            elif opps[r] == "ser":
                reset_time = randint(2,30)
                e = 0.8*torch.rand(1)
                delta = 0.5*torch.rand(1) 
                j_op = serJanken(delta = delta.item(),
                                reset_prob = 1/reset_time,
                                epsilon = e.item())

            elif opps[r] == "rnn" or opps[r] == "pucb":
                n = randint(0, len(weight_paths) -1)
                weights = torch.load(weight_paths[n], map_location = device)
                p = gruJanken()
                p.load_state_dict(weights["model_state_dict"])
                p.id = weights["id"]
                p.eval()
                if opps[r] == "pucb":
                    j_op = pucbJanken(p)
                else:
                    j_op = p
            
            elif opps[r] == "copy":
                epsilon = 0.2 + 0.8*torch.rand(1)
                j_op = copyJanken(epsilon = epsilon.item())

            print(f"\nEpoch {epoch+1} --- playing vs {j_op}")
            stats = []
            train(j, j_op, optimizer, err, 
                            device = device,
                            n_games = args.g,
                            stats = stats)

            with open(args.f, 'a') as f:
                f.write("Epoch {epoch + 1} vs. {j_op}\n")
                f.write( ','.join([format( x, ".3f") for x in stats]))
                f.write('\n')
            
        j.id += 1
        out_path = os.path.join( os.getcwd(), "weights", f"j_{j.id}.pt")
        weight_paths.append(out_path)
        torch.save( {"model_state_dict": j.state_dict(), "id": j.id}, out_path)

    torch.save( optimizer.state_dict(), optim_path)
