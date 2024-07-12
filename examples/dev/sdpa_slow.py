import torch
import torch.nn as nn
import torch.nn.functional as F
import thunder
import math

class ModelConfig:
    def __init__(self, n_embd=256, n_head=8, dropout=0.1, block_size=64, bias=True):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size

class Module(nn.Module):
    def __init__(self, config):
        """
        My implementation of NanoGPT Causal Self Attention module for PyTorch.

        Args:
        - config: Configuration object containing parameters for the attention module.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Forward pass of the Causal Self Attention module.

        Args:
        - x: Input tensor.

        Returns:
        - torch.Tensor: Output tensor after self-attention.
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

with torch.device('cuda'):
    config = ModelConfig(n_embd = 1536)
    module = Module(config)
    j_module = thunder.jit(module)

    batch_size, sequence_length, embedding_dim = 8, 16, config.n_embd
    x = torch.randn((batch_size, sequence_length, embedding_dim))

    ans = j_module(x)

    print(thunder.last_traces(j_module)[-1])



