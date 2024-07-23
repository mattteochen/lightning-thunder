import torch
import torch.nn as nn
import thunder

class ModelConfig:
    def __init__(self, n_embd=256, n_head=8, dropout=0.1, block_size=64, bias=True):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=torch.float32)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.float32)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

with torch.device('cuda'):
    with torch.cuda.amp.autocast():
        config = ModelConfig(n_embd = 3072)
        batch_size, sequence_length, embedding_dim = 16, 1024, config.n_embd
        x = torch.randn(batch_size, sequence_length, embedding_dim, dtype=torch.float32)

        model = CausalSelfAttention(config)
        jmodel_def = thunder.jit(model)
        jmodel_auto = thunder.jit(model, autotune_type='runtime')
        y = jmodel_def(x)
        y = jmodel_auto(x)

