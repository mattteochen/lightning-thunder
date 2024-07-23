import torch
import torch.nn as nn
import thunder

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y



# class ModelConfig:
#     def __init__(self, n_embd=256, n_head=8, dropout=0.1, block_size=64, bias=True):
#         self.n_embd = n_embd
#         self.n_head = n_head
#         self.dropout = dropout
#         self.bias = bias
#         self.block_size = block_size

# class CausalSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         # key, query, value projections for all heads, but in a batch
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=torch.float32)
#         # output projection
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.float32)
#         # regularization
#         self.attn_dropout = nn.Dropout(config.dropout)
#         self.resid_dropout = nn.Dropout(config.dropout)
#         self.n_head = config.n_head
#         self.n_embd = config.n_embd
#         self.dropout = config.dropout
#         # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
#         if not self.flash:
#             print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
#             # causal mask to ensure that attention is only applied to the left in the input sequence
#             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
#                                         .view(1, 1, config.block_size, config.block_size))

#     def forward(self, x):
#         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         if self.flash:
#             # efficient attention using Flash Attention CUDA kernels
#             y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
#         else:
#             # manual implementation of attention
#             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#             att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
#             att = F.softmax(att, dim=-1)
#             att = self.attn_dropout(att)
#             y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_dropout(self.c_proj(y))
#         return y

with torch.device('cuda'):
    num_heads = 8
    heads_per_dim = 64
    embed_dimension = num_heads * heads_per_dim
    dtype = torch.float16
    model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to(dtype)
    print(model)
    print(f'Training? {model.training}')
    batch_size = 32
    max_sequence_len = 256
    x = torch.randn(batch_size, max_sequence_len, embed_dimension, dtype=dtype)

    # config = ModelConfig(n_embd = 3072 // 2)
    # batch_size, sequence_length, embedding_dim = 16 // 2, 1024 // 2, config.n_embd

    # model = CausalSelfAttention(config)
    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime')
    y = jmodel_def(x)
    y = jmodel_auto(x)

