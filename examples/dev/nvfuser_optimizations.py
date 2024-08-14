import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_nvsight, torch_total_benchmark

class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.Linear(out_features, in_features),
            torch.nn.Linear(in_features, out_features),
            torch.nn.Linear(out_features, in_features)
        )
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        b = self.linear(x)
        c = b @ torch.transpose(b, 0, 1)
        for _ in range(4):
            c = c @ torch.transpose(c, 0, 1)
        return self.silu(c)

with torch.device('cuda'):
    in_features = 1 << 8
    out_features = 1 << 10
    model = Module(in_features, out_features)
    x = torch.randn(1 << 9, in_features, requires_grad=True)

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors=['nvfuser', 'cudnn', 'torch', 'python'])

    y = jmodel_def(x)
    y = jmodel_auto(x)

    iters = 100
    print('Results thunder benchmark:')
    traces = [
        thunder.last_traces(jmodel_def)[-1],
        thunder.last_traces(jmodel_auto)[-1],
        thunder.last_backward_traces(jmodel_def)[-1],
        thunder.last_backward_traces(jmodel_auto)[-1],
    ]
    labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
    thunder_fw_bw_benchmark(traces, labels, iters)
    thunder_fw_bw_benchmark(traces, labels, iters, nvsight=True)

    callables = [jmodel_def, jmodel_auto]
    labels = ['def', 'auto']
    inputs = [x, x]
    print('Results torch benchmark:')
    torch_fw_bw_benchmark(callables, labels, inputs, iters)
    torch_total_benchmark(callables, labels, inputs, iters)
    torch_fw_bw_benchmark_nvsight(callables, labels, inputs, iters)

    for t in traces:
        print(f'{t}\n#########################################')
