import torch
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.2

# Feedforward network
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed),
            nn.ReLU(),
            nn.Linear(n_embed,n_embed), #projection layer
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)