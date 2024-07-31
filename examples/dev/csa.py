import torch
import torch.nn as nn
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

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
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y

device = torch.device('cuda')
num_heads = 8
heads_per_dim = 64 * 1
embed_dimension = num_heads * heads_per_dim
dtype = torch.float16
model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to(device).to(dtype)
print(model)
batch_size = 1
max_sequence_len = 1024
x = torch.randn(batch_size, max_sequence_len, embed_dimension, dtype=dtype, requires_grad=True, device=device)

jmodel_def = thunder.jit(model)
jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'cudnn', 'torch', 'python'])

print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())


print('Results thunder benchmark:')
traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
thunder_fw_bw_benchmark(traces, labels, 50)

print('\n\nResults torch fw bw benchmark:')
callables = [jmodel_def, jmodel_auto]
labels = ['def', 'auto']
inputs = [x, x]
torch_fw_bw_benchmark(callables, labels, inputs, 50)

print('\n\n\n\n\n\n')
print(f'{thunder.last_traces(jmodel_def)[-1]}')
print('###############################################################################')
print(f'{thunder.last_traces(jmodel_auto)[-1]}')

print('\n\n')
print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
print('###############################################################################')
print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
