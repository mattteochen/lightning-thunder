import inspect
import torch
import thunder
import time

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
    a = 4096 * 1
    b = 11008 * 1
    x = torch.randn(2, 2048, a, requires_grad=True)

    jmodel_def = thunder.jit(LLaMAMLP(a, b), autotune_executors=False)
    jmodel_auto = thunder.jit(LLaMAMLP(a, b), autotune_executors=True)
    warm_up_iters = 2
    iters = 10

    for i in range(iters):
        start_fw = time.perf_counter_ns()
        y = jmodel_auto(x)
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
            print(f'tot time auto forward  = {(end_fw - start_fw) / 1000000} ms')
            print(f'tot time auto backward = {(end_bw - start_bw) / 1000000} ms')

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for i in range(iters):
        start_fw = time.perf_counter_ns()
        y = jmodel_def(x)
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
            print(f'tot time def forward  = {(end_fw - start_fw) / 1000000} ms')
            print(f'tot time def backward = {(end_bw - start_bw) / 1000000} ms')
    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
