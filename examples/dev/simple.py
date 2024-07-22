import torch
import thunder
from thunder.backend_optimizer.optimizer import benchmark_trace

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

    jmodel_def = thunder.jit(model, autotune_executors=False)
    jmodel_auto = thunder.jit(model, autotune_executors=True)
    stream = torch.cuda.current_stream()

    warm_up_iters = 2
    iters = 10
    for _ in range(warm_up_iters):
        y = jmodel_auto(x)
        yy = jmodel_def(x)
        grad_outputs = torch.ones_like(y)
        torch.autograd.grad(y, x, grad_outputs=grad_outputs)
        torch.autograd.grad(yy, x, grad_outputs=grad_outputs)

    print('\n\n')

    for i in range(1):

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        middle_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record(stream)
            y = jmodel_auto(x)
            middle_events[i].record(stream)
            torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

        torch.cuda.synchronize()
        fw = [s.elapsed_time(e) for s, e in zip(start_events, middle_events)]
        bw = [s.elapsed_time(e) for s, e in zip(middle_events, end_events)]
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        fw_time = sum(fw)
        bw_time = sum(bw)
        tot_time = sum(tot)
        print(f'Auto fw: {fw_time / iters}')
        print(f'Auto bw: {bw_time / iters}')
        print(f'Auto tot: {tot_time / iters}')
        print('\n')

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        middle_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record(stream)
            y = jmodel_def(x)
            middle_events[i].record(stream)
            torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

        torch.cuda.synchronize()
        fw = [s.elapsed_time(e) for s, e in zip(start_events, middle_events)]
        bw = [s.elapsed_time(e) for s, e in zip(middle_events, end_events)]
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        fw_time = sum(fw)
        bw_time = sum(bw)
        tot_time = sum(tot)
        print(f'Default fw: {fw_time / iters}')
        print(f'Default bw: {bw_time / iters}')
        print(f'Default tot: {tot_time / iters}')
        print('-------------------------------------------------------')

        c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False)
        print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False)
        print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False)
        print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False)
        print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')
        del o
    # print('\n\n\n\n\n\n')
    # print(f'{thunder.last_traces(jmodel_def)[-1]}')
    # print('###############################################################################')
    # print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    # print('\n\n')
    # print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    # print('###############################################################################')
    # print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("def"):
            y = jmodel_def(x)
            grad_outputs = torch.ones_like(y)
            torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("auto"):
            y = jmodel_auto(x)
            grad_outputs = torch.ones_like(y)
            torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
