import torch
import thunder
import time

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
    x = torch.randn(128, in_features)

    jmodel = thunder.jit(model, autotune_executors=True, executors=['nvfuser', 'torchcompile', 'torch'])

    for _ in range(10):
        start = time.perf_counter_ns()
        ans = jmodel(x)
        torch.autograd.grad(ans.sum(), model.parameters())
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        print(f'tot time = {(end - start) / 1000000} ms')

    print(thunder.last_backward_traces(jmodel)[-1])

