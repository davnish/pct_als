import torch
import torch.nn as nn
import torch.nn.functional as F

class pct(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers, n_labels = 8):
        super(pct, self).__init__()
        self.l1 = nn.Linear(3, n_embd)
               
        self.blocks = nn.Sequential(*[blocks(n_embd, n_heads) for _ in range(n_layers)])

        self.logits = nn.Linear(n_embd, n_labels)

    def forward(self, x):
        
        x = self.l1(x)

        x = self.blocks(x)

        logits = self.logits(x)
        
        return logits

class blocks(nn.Module):
    def __init__(self, n_embd, n_heads):
        super(blocks, self).__init__()
    
        head_size = n_embd // n_heads
        self.ln1 = nn.LayerNorm(n_embd)        
        self.multihead = multihead_attention(n_embd, head_size, n_heads)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffw = nn.Sequential(
                    nn.Linear(n_embd, n_embd * 4),
                    nn.ReLU(),
                    nn.Linear(n_embd * 4, n_embd),
        )
    def forward(self, x):
        multihead = x + self.multihead(self.ln1(x))
        return  multihead + self.ffw(self.ln2(multihead))

class multihead_attention(nn.Module):
    def __init__(self, n_embd, head_size, n_heads):
        super(multihead_attention, self).__init__()
        self.heads = nn.ModuleList([sa_layer(n_embd, head_size) for _ in range(n_heads)])
        self.ll = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], axis = -1)
        out = self.ll(out)
        
        return out

# Single Head
class sa_layer(nn.Module):
    def __init__(self, n_embd, head_size):
        
        super(sa_layer, self).__init__()
        self.q = nn.Linear(n_embd, head_size, bias = False)
        self.k = nn.Linear(n_embd, head_size, bias = False)
        self.v = nn.Linear(n_embd, head_size, bias = False)

    def forward(self, x):
    
        q = self.q(x)
        k = self.k(x)

        wei = q @ k.transpose(1,2) * (k.size(-1)  ** -0.5)
        wei = F.softmax(wei, dim = -1)

        v = self.v(x)

        out = wei @ v

        return out
    
