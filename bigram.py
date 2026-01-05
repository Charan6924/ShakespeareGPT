# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(1337)
from attention_head import MultiHeadAttention, Head
from feedforward import FeedForward
from residual_block import Block

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6

# Preprocessing
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join([itos[ix] for ix in l])
data = torch.tensor(encode(text), dtype = torch.long)

# Splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table = nn.Embedding(block_size,n_embed)
    self.blocks = nn.Sequential(
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        Block(n_embed,n_head),
        nn.LayerNorm((n_embed)))
    self.lm_head = nn.Linear(n_embed,vocab_size)

  def forward(self,idx,targets=None):
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    # (B,T,vocab_size)
    pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x)
    logits = self.lm_head(x) 

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits,loss = self(idx_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim = 1)
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx,idx_next),dim=1)
    return idx

# Instantiate model and optimizer
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training 
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))