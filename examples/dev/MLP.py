import torch
import torch.nn as nn
import thunder
from thunder.backend_optimizer.optimizer import benchmark_trace
# import logging

# torch._logging.set_logs(dynamo = logging.DEBUG)
# torch._dynamo.config.verbose = True

class ModelConfig:
    def __init__(self, n_embd=256, n_head=8, dropout=0.1, block_size=64, bias=True):
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

with torch.device('cuda'):
    embeddings = 3072
    config = ModelConfig(n_embd=embeddings)
    dtype = torch.float32
    x = torch.randn(16, 1024, embeddings, requires_grad=True)

    model = MLP(config)

    jmodel_def = thunder.jit(model)
    # This model fails under some circumstances after passed the placed traced under the rematelizer
    jmodel_auto = thunder.jit(model, autotune_type='emory', executors = ['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])

    y = model(x)
    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())
    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())

    print('########################################')
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_def_fw')
    print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_auto_fw')
    print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_def_bw')
    print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_auto_bw')
    print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')
    del o

    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("def"):
    #         y = jmodel_def(x)
    #         grad_outputs = torch.ones_like(y)
    #         torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # with profile(activities=[
    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("auto"):
    #         y = jmodel_auto(x)
    #         grad_outputs = torch.ones_like(y)
    #         torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

