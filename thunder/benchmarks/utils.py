from thunder.backend_optimizer.optimizer import benchmark_trace
import torch
import inspect
import time

def torch_fw_bw_benchmark_naive(models: list, torch_module: torch.nn.Module | None, labels: list, inputs: list, iters: int, int_input_tensor: bool = False) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(10):
            y = m(input)
            # Not supported by autograd
            if int_input_tensor:
                torch.autograd.grad(y.sum(), torch_module.parameters())
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))

        max_allocated_bytes = 0
        tot_time = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()

            s = time.time_ns()
            y = m(input)
            # Not supported by autograd
            if int_input_tensor:
                torch.autograd.grad(y.sum(), torch_module.parameters())
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))
            torch.cuda.synchronize()
            e = time.time_ns()
            tot_time += (e-s) / 1000000

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

            if i == iters-1:
                bw_func = inspect.getsource(y.grad_fn.compiled_backward)
                print(f'Compiled bw function:\n{bw_func}')

        tot_time = tot_time / iters
        print(f'{label} tot time: {tot_time / iters}')
        print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} GB')

def torch_fw_bw_benchmark(models: list, torch_module: torch.nn.Module | None, labels: list, inputs: list, iters: int, int_input_tensor: bool = False) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(10):
            y = m(input)
            # Not supported by autograd
            if int_input_tensor:
                torch.autograd.grad(y.sum(), torch_module.parameters())
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0

        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            start_events[i].record(stream)
            y = m(input)
            # Not supported by autograd
            if int_input_tensor:
                torch.autograd.grad(y.sum(), torch_module.parameters())
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

            if i == iters-1:
                bw_func = inspect.getsource(y.grad_fn.compiled_backward)
                print(f'Compiled bw function:\n{bw_func}')

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot time: {tot_time / iters}')
        print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} GB')


def thunder_fw_bw_benchmark(traces: list, labels: list, iters: int) -> None:
    for trc, label in zip(traces, labels):
        c, m, _ = benchmark_trace(trc, apply_del_last_used=False, snapshot=True, snapshot_name=label, iters=iters)
        print(f'Executing {label} trace:\n{c} ms, {m / (2**30)} GB')

