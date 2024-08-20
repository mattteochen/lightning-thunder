import torch
import thunder
from thunder.backend_optimizer.optimizer import benchmark_trace

torch.set_default_dtype(torch.float32)

class Model(torch.nn.Module):
    def __init__(self, inf, outf) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(inf, outf, bias=False)

    def forward(self, query, key, value):
        query = self.linear(query)
        a = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return a

with torch.device('cuda'):
    features = 128
    model = Model(features, features)

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors=['nvfuser', 'cudnn', 'sdpa', 'fa3', 'torchcompile'])

    q = torch.rand(32, 8, 128, features, requires_grad=True)
    k = torch.rand(32, 8, 128, features, requires_grad=True)
    v = torch.rand(32, 8, 128, features, requires_grad=True)

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




