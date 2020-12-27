#!/usr/bin/env python3
import argparse
from janken import *
from random import randint, choice, uniform
from glob import glob
import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

def train(bot, bot_op, optimizer, err, **kwargs):
    interval = kwargs.get("interval", 10)
    n_it = kwargs.get("n_it", 128)
    device = kwargs.get("device")    
    n_games = kwargs.get("n_games", 5)
    dataset = kwargs.get("data")
    stats = kwargs.get("stats")
  
    for game in range(n_games):
        bot.reset()
        bot_op.reset()

        rewards = []
        hidden_states = []
        targets = []
        moves = []
        opp_moves = []
        norm_reward = 0.
        for _ in range(n_it):
            for i in range(interval):
                m1 = bot.throw()
                m2 = bot_op.throw()
                
                reward = WIN[m1,m2]
                rewards.append(reward - optim_reward(bot_op.dist)) 
                bot.observe(m1, reward, device)
                if isinstance(bot_op, rnnJanken) or isinstance(bot_op, pucbJanken):
                    bot_op.observe(m2, reward, device)
                else:
                    bot_op.observe(m2, -reward)
                
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
        
        #save data for later
        dataset["states"].append(hidden_states.data)
        dataset["inputs"].append(encoded.data)
        dataset["targets"].append(targets.data)

        loss = err(outputs, targets)
        
        avg_reward = sum(rewards)/len(rewards)
        loss *= -avg_reward.item()
        loss = loss.to(device)
        #backpropagate
        optimizer.zero_grad()
        loss.backward(retain_graph = False if game == n_games -1 else True)
        optimizer.step()
        if stats is not None:
            stats.append(avg_reward)

        print(f"reward: {avg_reward.item():.3f}\tloss: {loss.item():.3f}")
   

def replay_train(bot, optimizer, err, **kwargs):
        device = kwargs.get("device")
        dataset = kwargs.get("data")
        n_batch = kwargs.get("n_batch")
        epochs = kwargs.get("epochs",1)
        print("Replay")
        if isinstance(data, TensorDataset):
            loader = DataLoader( data, batch_size = 64,
                                shufffle = True,
                                pin_memory = True)
            for e in range(epoch):
                for i, data in enumerate(loader):
                    running_loss = 0.
                    inputs, states, targets = data
                    inputs = inputs.to(device)
                    states = torch.transpose(states, 0,1).to(device)
                    targets = targets.to(device)
                    out, _ = bot(inputs, hidden_states)

                    loss = err(out, targets)
                    running_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i%100 == 99:
                        print(f"loss: {running_loss/100:.3f}")
                        running_loss = 0.
             
        if n_batch <= 0:
            return
        running_loss = 0.
        for _ in range(n_batch):
            #replay random game
            i = randint(0, len(dataset["inputs"])-1)
            inputs = dataset["inputs"][i]
            targets = dataset["targets"][i]
            hidden_states = dataset["states"][i] 
            inputs, targets, hidden_states = inputs.to(device), targets.to(device), hidden_states.to(device)
            outputs, _ = bot(inputs, hidden_states)
            loss = err( outputs, targets)
            running_loss += loss
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
        print(f"loss: {running_loss/n_batch:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "training script for Janken bot")
    parser.add_argument("-n", help = "number of opponents per epoch", type = int, dest = 'n', default = 10)
    parser.add_argument("-g", help = "number of games per opponent", type = int, dest = 'g', default = 10)
    parser.add_argument("-e", help = "number of epochs", type = int, dest = 'e', required = True)
    parser.add_argument("-f", help = "file to write stats", type = str, dest = 'f',\
                              default = os.path.join( os.getcwd(), "stats.txt"))
    parser.add_argument("--data", type = str, dest = "data")
    parser.add_argument("--replay", action = "store_true", dest = "replay")
    args = parser.parse_args()    
    device = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available else "cpu")
 
    j = rnnJanken(model_type = "GRU")
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
    if args.data:
        dataset = torch.load(args.data, map_location = device)
    else:
        dataset = {"inputs":[], "states":[], "targets":[]}

    if args.replay:
        targs = torch.cat(data["targets"])
        states = torch.cat(data["states"], dim = 1)
        states = torch.transpose(states, 0,1)
        inputs = torch.cat(data["inputs"]).squeeze(1)
        tensor_data = TensorDataset(inps, states, targs)
        replay_train(j, optimizer, err,
                    device = device,
                    data = tensor_data)
    else: 
        for epoch in range(args.e):
            for opp_round in range(args.n):
                #choose the opponent and random hyperparameters
                opps = ["const", "rand", "copy", "exp3r",
                        "ucb", "unif", "bayes","rnn"]
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
                    p = rnnJanken(model_type = "GRU", epsilon = 0.2)
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
                    j_op = bayesJanken(gamma = uniform(0,0.8))

                print(f"\nEpoch {epoch+1} --- playing vs {j_op}")
                stats = []
                j.reset()

                try:
                    train(j, j_op, optimizer, err, 
                                    device = device,
                                    n_games = args.g,
                                    data = dataset,
                                    stats = stats)
                except KeyboardInterrupt:
                    s = input("save data? (y/n)")
                    if s == "y":
                        torch.save(dataset, f"data_{j.id}.pt")
                    sys.exit()

                replay_train(j, optimizer, err,
                            data = dataset, 
                            n_batch = (opp_round)*args.g, 
                            device = device)

                with open(args.f, 'a') as f:
                    f.write(f"Epoch {epoch + 1} vs. {j_op}\n")
                    f.write( ','.join([format( x, ".3f") for x in stats]))
                    f.write('\n')
                
            j.id += 1
            out_path = os.path.join( os.getcwd(), "weights", f"j_{j.id}.pt")
            weight_paths.append(out_path)
        torch.save(dataset, f"data_{j.id}.pt")

    torch.save( {"model_state_dict": j.state_dict(), "id": j.id}, out_path)
    torch.save( optimizer.state_dict(), optim_path)
