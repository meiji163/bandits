import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

# history: 128 agent moves / 128 opponent moves
# 100x3x2 --> 3 (policy)

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1, use_1x1 = False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size = 3,
                                stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size =3, padding =1, bias = False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        if use_1x1:
            self.conv3 = nn.Conv1d(in_planes, out_planes, kernel_size = 1,
                                    stride = stride, bias = False)
        else:
            self.conv3 = None

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2( res ))
        if self.conv3:
            x = self.conv3(x)
        x += res
        return self.relu(x)

def make_block(in_planes, out_planes, block_size, first_block = False):
    block = []
    for i in range(block_size):
        if i == 0 and not first_block:
            block.append(ResBlock(in_planes, out_planes, use_1x1 = True, stride = 2))
        else:
            block.append(ResBlock(in_planes, out_planes))
        return nn.Sequential(*block)

class JankenNet(nn.Module):
    def __init__(self):
        super(JankenNet, self).__init__()
        self.blk1 = nn.Sequential(nn.Conv1d(6,64, kernel_size = 7, stride = 2, padding = 3),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1))
        self.blk2 = make_block(64,64,2, first_block = True)
        self.blk3 = make_block(64,128,2)
        self.blk4 = make_block(128,256,2)
        self.pool = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)
        self.drop = nn.Dropout(p = 0.2)
        self.lin = nn.Linear(256,3)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.pool(self.drop(x))
        x = x.view(-1)
        x = self.lin(x)
        return x

SOFT = nn.Softmax(dim = 0)

class JankenBot():
    def __init__(self, policy: JankenNet):
        self.policy = policy
        self.history = torch.zeros(6,256)

    def throw(self, dist = False):
        d = self.dist
        if dist:
            return d.sample(), d
        return d.sample()

    def update(self, a, b):
        self.history[:,:-1] = self.history[:,1:]
        self.history[a,-1] = 1
        self.history[b+3, -1] = 1
        
    @property
    def dist(self):
        out = self.policy(self.history.unsqueeze(0))
        return Categorical( SOFT(out).squeeze(0))
