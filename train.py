from janken import *
from random import randint
from glob import glob
import os
import torch

def train(bot, bot_op, optimizer, **kwargs):
    interval = kwargs.get("interval", 100)
    n_it = kwargs.get("n_it", 10)
    
    avg_rewards = []
    for _ in range(n_it):
        dists = []
        op_dists = []
        for i in range(interval):
            dists.append(bot.dist)
            op_dists.append(bot_op.dist)

            m1 = bot.throw()
            m2 = bot_op.throw()
            reward = WIN[m1,m2]
            bot.observe(m1, reward)
            bot_op.observe(m2, -reward)
        
        avg_reward = 0.
        for i in range(len(dists)):
            avg_reward += expected_reward( dists[i], op_dists[i] )
        avg_reward /= interval
        loss = -avg_reward
        avg_rewards.append(avg_reward.item())

        #backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return avg_rewards

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "training script for Janken bot")
    parser.add_argument("-i", help = "update interval", type = int, dest = 'i', default = 20)
    parser.add_argument("-n", help = "number of intervals", type = int, dest = 'n', default = 500)
    parser.add_argument("-e", help = "number of epochs", type = int, dest = 'e')
    parser.add_argument("-f", help = "file to write stats", type = str, dest = 'f',\
                              default = os.path.join( os.getcwd(), "stats.txt"))

    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available else "cpu")
    j = JankenRNN(device)
    j.to(device)

    optimizer = torch.optim.AdamW(j.parameters(), lr = 1e-3)
   
    games = 10 #number of games to play against each opponent 
    for epoch in range(args.e):
        #choose the opponent and random hyperparameters
        opps = ["dumb", "exp3r", "ucb", "rnn"]
        r = randint(0, len(opps)-1)
        if opps[r] == "dumb":
            reset_time = randint(2,1000)
            b = 0.5*torch.rand(1).item()
            j_op = dumbJanken(bias = b, reset_prob = 1/reset_time)

        elif opps[r] == "exp3r":
            exploration = 0.3 + 0.2*torch.randn(1)
            H = randint(100,400)
            j_op = exp3rJanken(gamma = exploration.item(), H = H)

        elif opps[r] == "ucb":
            exploration = 5*torch.rand(1)
            e = 0.8*torch.rand(1)
            j_op = ucbJanken(gamma = exploration.item(),
                             epsilon = e.item())

        elif opps[r] == "rnn":
            weight_paths = glob("weights/j*.pt")    
            n = randint(0, len(weight_paths) - 1)
            weights = torch.load(weight_paths[n], map_location = device)
            j_op = JankenRNN()
            j_op.load_state_dict(weights["model_state_dict"])
            j_op.id = weights["id"]
            j_op.to(device)

        stats = []            
        for _ in range(games):
            avg = train(j, j_op, optimizer, 
                        interval = args.i, 
                        n_it = args.n)
            stats.extend(avg)
        
        with open(args.f, 'a') as f:
            f.write(f"Epoch {epoch + 1} --- {j_op}\n")
            f.write(",".join([format(x, '.3f') for x in stats]))                    

        j.id += 1
        out_path = os.path.join( os.getcwd(), "weights", f"j_{j.id}.pt")
        torch.save( {"model_state_dict": j.state_dict(), "id": j.id}, out_path)
