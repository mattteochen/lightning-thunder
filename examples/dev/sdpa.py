import torch
import thunder
from thunder.backend_optimizer.optimizer import benchmark_trace

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

    print('########################################')
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='sdpa_def_fw', iters=10)
    print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='sdpa_auto_fw', iters=10)
    print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='sdpa_def_bw', iters=10)
    print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='sdpa_auto_bw', iters=10)
    print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')

    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')




