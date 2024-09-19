"""
This benchmark script is intended to demonstrate the autotuner on a generic model.
No executor are given leaving full responsibility to Thunder.
"""

import torch
import thunder
from thunder.benchmarks.utils import torch_timer_total_benchmark, torch_total_benchmark


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
    mult = 2
    a = 4096 * mult
    b = 11008 * mult
    x = torch.randn(4, 2048, a, requires_grad=True)

    model = LLaMAMLP(a, b)

    eager = model
    torchcompile = torch.compile(model)
    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(
        model,
        autotune_type="runtime",
        autotune_enable_te=True,
        autotune_nv_enable_options=True,
        model_name="LLaMAMLP",
        autotune_save_configuration=True,
    )

    print("deviation def:", (jmodel_def(x) - model(x)).abs().max().item())
    print("deviation auto:", (jmodel_auto(x) - model(x)).abs().max().item())

    iters = 100
    callables = [eager, torchcompile, jmodel_def, jmodel_auto]
    labels = ["eager", "torchcompile", "Thunder", "Thunder Autotuned"]
    inputs = [x, x, x, x]
    print("\nResults with torch total benchmark:")
    torch_total_benchmark(callables, labels, inputs, iters)
    print("\nResults with torch timer benchmark:")
    torch_timer_total_benchmark(callables, labels, inputs, "LlamaMLP")
