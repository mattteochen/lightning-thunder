import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark

class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear_a = torch.nn.Linear(in_features, out_features)
        self.linear_b = torch.nn.Linear(out_features, in_features)
        self.linear_c = torch.nn.Linear(in_features, out_features)
        self.linear_d = torch.nn.Linear(out_features, in_features)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        b = self.linear_d(self.linear_c(self.linear_b(self.linear_a(x))))
        c = b @ torch.transpose(b, 0, 1)
        for _ in range(10):
            c = c @ torch.transpose(c, 0, 1)
        return self.silu(c)

with torch.device('cuda'):
    in_features = 1 << 8
    out_features = 1 << 10
    model = Module(in_features, out_features)
    x = torch.randn(1 << 9, in_features, requires_grad=True)

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors=['nvfuser', 'cudnn', 'torch', 'python'], )

    y = jmodel_def(x)
    y = jmodel_auto(x)

    print('Results thunder benchmark:')
    traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
    labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
    thunder_fw_bw_benchmark(traces, labels, 50)

    callables = [jmodel_def, jmodel_auto]
    labels = ['def', 'auto']
    inputs = [x, x]
    print('Results torch benchmark:')
    torch_fw_bw_benchmark(callables, labels, inputs, 50)

    for t in traces:
        print(f'{t}\n#########################################')
