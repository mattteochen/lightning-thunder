import torch
import torch.nn as nn
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_nvsight

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
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'sdpa', 'torch', 'python'])

    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())

    print('Results with thunder benchmark:')
    traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
    labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
    thunder_fw_bw_benchmark(traces, labels, 50, nvsight = False)

    callables = [jmodel_def, jmodel_auto]
    labels = ['def', 'auto']
    inputs = [x, x]
    print('Results with torch fw bw benchmark:')
    torch_fw_bw_benchmark(callables, labels, inputs, 50)

    # for t in traces:
    #     print(t)
    #     print('##########################')

