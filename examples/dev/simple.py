import torch
import thunder
import time
import inspect

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

    jmodel_default = thunder.jit(Module(in_features, out_features), autotune_executors=False)
    jmodel_autotune = thunder.jit(Module(in_features, out_features), autotune_executors=True)
    x = torch.randn(128, in_features, requires_grad=True)
    warm_up_iters = 3
    for i in range(10):
        start_fw = time.perf_counter_ns()
        y = jmodel_default(x)
        torch.cuda.synchronize()
        end_fw = time.perf_counter_ns()
        grad_outputs = torch.ones_like(y)
        torch.cuda.synchronize()
        start_bw = time.perf_counter_ns()
        torch.autograd.grad(y, x, grad_outputs=grad_outputs)
        torch.cuda.synchronize()
        end_bw = time.perf_counter_ns()
        torch.cuda.empty_cache()
        # source = inspect.getsource(y.grad_fn.compiled_backward)

        if i >= warm_up_iters:
            print(f'tot time default forward  = {(end_fw - start_fw) / 1000000} ms')
            print(f'tot time default backward = {(end_bw - start_bw) / 1000000} ms')

    for i in range(10):
        start_fw = time.perf_counter_ns()
        y = jmodel_autotune(x)
        torch.cuda.synchronize()
        end_fw = time.perf_counter_ns()
        grad_outputs = torch.ones_like(y)
        torch.cuda.synchronize()
        start_bw = time.perf_counter_ns()
        torch.autograd.grad(y, x, grad_outputs=grad_outputs)
        torch.cuda.synchronize()
        end_bw = time.perf_counter_ns()
        torch.cuda.empty_cache()
        # source = inspect.getsource(y.grad_fn.compiled_backward)

        if i >= warm_up_iters:
            print(f'tot time autotune forward  = {(end_fw - start_fw) / 1000000} ms')
            print(f'tot time autotune backward = {(end_bw - start_bw) / 1000000} ms')
    # print('\n\n', thunder.last_backward_traces(jmodel)[-1])

