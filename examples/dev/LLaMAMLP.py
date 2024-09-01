"""
This benchmark script is intended to demonstrate the optimizer on a generic model.
No executor are given leaving full responsibility to the engine.
"""

import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_total_benchmark


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


with torch.device("cuda"):
    mult = 1
    a = 4096 * mult
    b = 11008 * mult
    x = torch.randn(2, 2048, a, requires_grad=True)

    model = LLaMAMLP(a, b)

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type="runtime", autotune_enable_te=True)

    print("deviation def:", (jmodel_def(x) - model(x)).abs().max().item())
    print("deviation auto:", (jmodel_auto(x) - model(x)).abs().max().item())

    iters = 100
    print("Results with thunder benchmark:")
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
    labels = ["def", "auto"]
    inputs = [x, x]
    print("\nResults with torch fw bw benchmark:")
    torch_fw_bw_benchmark(callables, labels, inputs, iters)
    print("\nResults with torch total benchmark:")
    torch_total_benchmark(callables, labels, inputs, iters)
