import torch
import thunder

class LLaMAMLP(torch.nn.Module):
    def __init__(self, n_embd, intermediate_size) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(n_embd, intermediate_size, bias=False)
        self.fc_2 = torch.nn.Linear(n_embd, intermediate_size, bias=False)
        self.proj = torch.nn.Linear(intermediate_size, n_embd, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

with torch.device('cuda'):
    from thunder.backend_optimizer.optimizer import benchmark_trace
    # See changes from mult = 1 to mult = 4
    mult = 4
    a = 4096 * mult
    b = 11008 * mult
    x = torch.randn(2, 2048, a, requires_grad=True)
    model = LLaMAMLP(a, b)
    jmodel_def = thunder.jit(model, executors=['torchcompile', 'nvfuser'])
    jmodel_auto = thunder.jit(model, autotune_type='runtime')

    y = model(x)
    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())
    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())

    print('########################################')
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='LLaMAMLP_def_fw')
    print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='LLaMAMLP_auto_fw')
    print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='LLaMAMLP_def_bw')
    print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='LLaMAMLP_auto_bw')
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
