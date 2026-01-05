import torch
import torch.nn as nn
import torch.nn.functional as F

n_embed = 384
block_size = 256
dropout = 0.2

class Head(nn.Module):
    def __init__(self,head_size):
      super().__init__()
      self.key = nn.Linear(n_embed,head_size,bias=False)
      self.query = nn.Linear(n_embed,head_size,bias=False)
      self.value = nn.Linear(n_embed,head_size,bias=False)
      self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
      self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)

      wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
      wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))  # type: ignore
      wei = F.softmax(wei,dim=-1)
      wei = self.dropout(wei)
      v = self.value(x)
      out = wei @ v
      return out
    
# Multiple attentions
class MultiHeadAttention(nn.Module):
    """multiple attention heads in parallel"""
    def __init__(self,num_heads,head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(num_heads * head_size,n_embed)
      self.dropout = nn.Dropout(dropout)

    def forward(self,x):
       out = torch.cat([head(x) for head in self.heads],dim = -1)
       out = self.proj(out)
       return out