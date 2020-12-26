#!/usr/bin/env python3
import argparse
from janken import *
from random import randint, choice, uniform
from glob import glob
import os
import torch

def train(bot, bot_op, optimizer, err, **kwargs):
    interval = kwargs.get("interval", 10)
    n_it = kwargs.get("n_it", 128)
    device = kwargs.get("device")    
    n_games = kwargs.get("n_games", 5)
    stats = kwargs.get("stats")
  
    for game in range(n_games):
        bot.reset()
        bot_op.reset()
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
                bot.observe(m1, reward, device)
                if isinstance(bot_op, rnnJanken) or isinstance(bot_op, pucbJanken):
                    bot_op.observe(m2, reward, device)
                else:
                    bot_op.observe(m2, -reward)
                
                rewards.append(reward.item() - max_reward(bot_op.dist).item())
                if i == interval-1:
                    moves.append( m1.unsqueeze(0))
                    opp_moves.append( m2.unsqueeze(0))
                    hidden_states.append( bot.hidden_state)
                    targ = bot_op.throw().item()
                    targets.append(targ)
        
        inputs = (torch.tensor(opp_moves).unsqueeze(0).T,
                 torch.tensor(moves).unsqueeze(0).T)
        targets = torch.tensor(targets).to(device)
        hidden_states = torch.cat( hidden_states, dim = 1).to(device)
        encoded = bot.encode(*inputs, device)
        outputs, _ = bot(encoded, hidden_states)
        loss = err(outputs, targets)
        
        avg_reward = sum(rewards)/len(rewards)
        loss *= -(avg_reward 
        #backpropagate
        optimizer.zero_grad()
        loss.backward(retain_graph = False if game == n_games -1 else True)
        optimizer.step()
        if stats is not None:
            stats.append(avg_reward)

        print(f"{sum(rewards):.0f}/{len(rewards)}\t loss: {loss.item():.3f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "training script for Janken bot")
    parser.add_argument("-n", help = "number of opponents per epoch", type = int, dest = 'n', default = 10)
    parser.add_argument("-g", help = "number of games per opponent", type = int, dest = 'g', default = 10)
    parser.add_argument("-e", help = "number of epochs", type = int, dest = 'e', required = True)
    parser.add_argument("-f", help = "file to write stats", type = str, dest = 'f',\
                              default = os.path.join( os.getcwd(), "stats.txt"))
    args = parser.parse_args()    

    device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available else "cpu")
 
    j = lstmJanken()
    j.train()
    j.to(device)
    weight_paths = glob("weights/j_*.pt")

    if len(weight_paths) == 0:
        torch.save( {"model_state_dict": j.state_dict(), "id": 0}, os.path.join("weights", "j_0.pt"))
    else:
        
        def get_id(path):
            match = re.match(r"weights/j_(\d+).pt", path)
            return int(match[1])

        latest = max(weight_paths, key = get_id) 
        weight = torch.load( latest, map_location = device)
        print(f"Loading {latest}")
        j.load_state_dict(weight["model_state_dict"])
        j.id = weight["id"]
    
    err = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(j.parameters(), lr = 1.0)
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
            opps = ["const", "rand", "copy", "exp3r", "ucb", "unif", "ser"]+ 3*["pucb","rnn"]
            opp = choice(opps) 

            if opp == "rand":
                reset_time = randint(2,100)
                b = 0.8*torch.rand(1).item()
                j_op = randJanken(bias = b, reset_prob = 1/reset_time)

            elif opp == "const":
                prob = torch.rand(1) 
                j_op = constJanken(reset_prob = prob) 

            elif opp == "unif":
                j_op = randJanken(bias = 0, reset_prob = 0)

            elif opp == "exp3r":
                exploration = uniform(0.1, 0.4)
                H = randint(100,300)
                j_op = exp3rJanken(gamma = exploration, H = H)

            elif opp == "ucb":
                reset_time = randint(2,30)
                exploration = uniform(0,4) 
                e = uniform(0, 0.6) 
                j_op = ucbJanken(gamma = exploration,
                                 epsilon = e,
                                 reset_prob = 1/reset_time)

            elif opp == "ser":
                prob = uniform(0,1) 
                e = uniform(0.2,0.8) 
                delta = uniform(0, 0.5) 
                j_op = serJanken(delta = delta,
                                reset_prob = prob, 
                                epsilon = e)

            elif opp == "rnn" or opp == "pucb":
                opp_weight = choice(weight_paths)
                weights = torch.load(opp_weight, map_location = device)
                p = lstmJanken()
                p.load_state_dict(weights["model_state_dict"])
                p.id = weights["id"]
                p.to(device)
                p.eval()
                if opp == "pucb":
                    prob = uniform(0.2,1)
                    exploration = uniform(0,4)
                    e = uniform(0,0.5)
                    j_op = pucbJanken(p, gamma = exploration,
                                 epsilon = e,
                                 reset_prob = prob) 
                else:
                    j_op = p
            
            elif opp == "copy":
                e = uniform(0.2,1) 
                j_op = copyJanken(epsilon = e)
        
            elif opp == "bayes":
                prob = uniform(0,0.6)
                j_op = bayesJanken(reset_prob = prob)


            print(f"\nEpoch {epoch+1} --- playing vs {j_op}")
            stats = []
            j.reset()
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
