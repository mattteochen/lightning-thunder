import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query, key, value):
        a = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return a

with torch.device('cuda'):
    model = Model()

    jmodel_def = thunder.jit(model)
    # Order does not matter anymore
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])

    q = torch.rand(32, 8, 128, 64*16, dtype=torch.float32, requires_grad=True)
    k = torch.rand(32, 8, 128, 64*16, dtype=torch.float32, requires_grad=True)
    v = torch.rand(32, 8, 128, 64*16, dtype=torch.float32, requires_grad=True)

    print('deviation def:', (jmodel_def(q, k, v) - model(q, k, v)).abs().max().item())
    print('deviation auto:', (jmodel_auto(q, k, v) - model(q, k, v)).abs().max().item())

    iters = 100
    fw_traces = [
        thunder.last_traces(jmodel_def)[-1],
        thunder.last_traces(jmodel_auto)[-1],
    ]
    bw_traces = [
        thunder.last_backward_traces(jmodel_def)[-1],
        thunder.last_backward_traces(jmodel_auto)[-1],
    ]
    fw_labels = ["fw_def", "fw_auto"]
    bw_labels = ["bw_def", "bw_auto"]
    thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters)

    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')




