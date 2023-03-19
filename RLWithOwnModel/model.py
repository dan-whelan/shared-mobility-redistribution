import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, outputs):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(state_dim, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, outputs)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x