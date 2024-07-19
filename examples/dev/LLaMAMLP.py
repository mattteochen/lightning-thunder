import torch
import thunder

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
    from thunder.backend_optimizer.optimizer import OptimizerType, benchmark_trace
    a = 4096 * 1
    b = 11008 * 1
    x = torch.randn(2, 2048, a, requires_grad=True)

    jmodel_def = thunder.jit(LLaMAMLP(a, b))
    jmodel_auto = thunder.jit(LLaMAMLP(a, b), autotune_type='memory')
    warm_up_iters = 2
    iters = 10
    stream = torch.cuda.current_stream()

    for _ in range(warm_up_iters):
        y = jmodel_auto(x)
        yy = jmodel_def(x)
        torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
        torch.autograd.grad(yy, x, grad_outputs=torch.ones_like(y))

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

        c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='def_fw')
        print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='auto_fw')
        print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='def_bw')
        print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='auto_bw')
        print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')
        del o

    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

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
