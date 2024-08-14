import torch
from thunder.backend_optimizer.utils import benchmark_trace

warm_up_iters = 50

def torch_fw_bw_benchmark_nvsight(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            y = m(input)
            y.sum().backward()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda.nvtx.range_push(f"{label}: fw-bw iter {i}")
            y = m(input)
            y.sum().backward()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            y = m(input)
            y.sum().backward()

        torch.cuda.synchronize()
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
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot fw time: {tot_time} ms')
        print(f'{label} max fw allocated memory: {max_allocated_bytes / (2**30)} GB')

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            y = m(input)
            start_events[i].record(stream)
            y.sum().backward()
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot bw time: {tot_time} ms')
        print(f'{label} max bw allocated memory: {max_allocated_bytes / (2**30)} GB')

def torch_total_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            y = m(input)
            y.sum().backward()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            start_events[i].record(stream)
            y = m(input)
            y.sum().backward()
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot time: {tot_time} ms')
        print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} GB')


def thunder_fw_bw_benchmark(traces: list, labels: list, iters: int, nvsight: bool = False) -> None:
    for trc, label in zip(traces, labels):
        c, m, _ = benchmark_trace(trc, apply_del_last_used=False, snapshot=True, snapshot_name=label, iters=iters, nvsight=nvsight, nvsight_fn_name=label)
        print(f'Executing {label} trace:\n{c} ms, {m / (2**30)} GB')

