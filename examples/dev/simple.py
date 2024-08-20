import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark

class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        a = x + x
        b: torch.Tensor = self.linear(a)
        c = b * b
        return self.silu(c)

with torch.device('cuda'):
    in_features = 4096
    out_features = 11008
    model = Module(in_features, out_features)
    x = torch.randn(128, in_features, requires_grad=True)

    jmodel_def = thunder.jit(model, )
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors=['nvfuser', 'torchcompile', 'cudnn', 'torch', 'python'], )

    y = jmodel_def(x)
    y = jmodel_auto(x)

    iters = 100
    print('Results thunder benchmark:')
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

    callables = [jmodel_def, jmodel_auto]
    labels = ['def', 'auto']
    inputs = [x, x]
    print('Results torch benchmark:')
    torch_fw_bw_benchmark(callables, labels, inputs, 50)
