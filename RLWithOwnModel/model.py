import torch 
import torch.nn as nn
import torch.nn.functional as Functional

class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(500, 4)
        self.lin1 = nn.Linear(4, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, outputs)
    
    def forward(self, x):
        x = Functional.relu(self.lin1(self.embedding(x)))
        x = Functional.relu(self.lin2(x))
        x = self.lin3(x)
        return x
