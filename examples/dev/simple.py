import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_naive

class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        a = x + x
        b: torch.Tensor = self.linear(a)
        c = b * b
        d = c + c
        return self.silu(d)

with torch.device('cuda'):
    multiplier = 1000
    in_features = 20 * multiplier
    out_features = 30 * multiplier
    model = Module(in_features, out_features)
    x = torch.randn(128, in_features, requires_grad=True)

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime')

    y = jmodel_def(x)
    y = jmodel_auto(x)

    print('Results thunder benchmark:')
    traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
    labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
    thunder_fw_bw_benchmark(traces, labels, 10)

    callables = [jmodel_def, jmodel_auto]
    labels = ['def', 'auto']
    inputs = [x, x]
    print('Results torch benchmark:')
    torch_fw_bw_benchmark(callables, model, labels, inputs, 10)
    print('Results torch benchmark naive:')
    torch_fw_bw_benchmark_naive(callables, model, labels, inputs, 10)

    for t in traces:
        print(t)
        print('####################################')
