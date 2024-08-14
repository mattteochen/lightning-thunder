import torch
import torch.nn as nn
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_nvsight, torch_total_benchmark

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
    config = ModelConfig(n_embd=embeddings, dropout=0.0, bias=False)
    dtype = torch.float32
    x = torch.randn(16, 1024, embeddings, requires_grad=True)

    model = MLP(config)

    jmodel_def = thunder.jit(model)
    # This model fails under some circumstances after passed the placed traced under the rematelizer
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'sdpa', 'torch', 'python'], use_cudagraphs=False)

    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())

    iters = 100
    callables = [jmodel_auto, jmodel_def]
    labels = ['auto', 'def']
    inputs = [x, x]
    print('Results with torch fw bw benchmark:')
    torch_fw_bw_benchmark(callables, labels, inputs, iters)
    torch_total_benchmark(callables, labels, inputs, iters)
    torch_fw_bw_benchmark_nvsight(callables, labels, inputs, iters)

    print('Results with thunder benchmark:')
    traces = [
        thunder.last_traces(jmodel_def)[-1],
        thunder.last_traces(jmodel_auto)[-1],
        thunder.last_backward_traces(jmodel_def)[-1],
        thunder.last_backward_traces(jmodel_auto)[-1],
    ]
    traces.reverse()
    labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
    labels.reverse()
    thunder_fw_bw_benchmark(traces, labels, iters, nvsight = False)
    thunder_fw_bw_benchmark(traces, labels, iters, nvsight = True)

    # for t in traces:
    #     print(t)
    #     print('##########################')

