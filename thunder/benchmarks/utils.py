from collections.abc import Callable, Sequence
import torch
from thunder.backend_optimizer.utils import benchmark_trace
from thunder.core.trace import TraceCtx

warm_up_iters = 50

class SplitFwBwBenchmarkUtils():
    def __init__(
            self, *, cost: float = float("inf"), fw_fn: Callable | None = None, bw_fn: Callable | None = None, executor = None
    ) -> None:
        self.cost: float = cost
        self.fw_fn: Callable | None = fw_fn
        self.bw_fn: Callable | None = bw_fn
        self.executor = executor

class AutotunerTorchAutogradBenchmarkUtils():
    def __init__(
        self,
        cost: float = float('inf'),
        fw_trace: TraceCtx | None = None,
        bw_trace: TraceCtx | None = None,
        fw_traces: Sequence[TraceCtx] = [],
        bw_traces: Sequence[TraceCtx] = [],
        primal_trace: TraceCtx | None = None,
        executor = None,
        selected_executors: Sequence = []
        ) -> None:
        self.cost: float = cost
        self.fw_trace = fw_trace
        self.bw_trace = bw_trace
        self.fw_traces = fw_traces
        self.bw_traces = bw_traces
        self.primal_trace = primal_trace
        self.executor = executor
        self.selected_executors = selected_executors


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
            torch.cuda.nvtx.range_push(f"torch training {label}, iter {i}")
            torch.cuda.nvtx.range_push('forward')
            y = m(input)
            torch.cuda.nvtx.range_pop()
            loss = y.sum()
            torch.cuda.nvtx.range_push('backward')
            loss.backward()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            y = m(input)
            y.sum().backward()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        torch.cuda.synchronize()
        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

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

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        torch.cuda.synchronize()
        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            y = m(input)
            loss = y.sum()
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            start_events[i].record(stream)
            loss.backward()
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
            y = m(*input if isinstance(input, tuple) else input)
            y.sum().backward()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        torch.cuda.synchronize()
        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

            start_events[i].record(stream)
            y = m(*input if isinstance(input, tuple) else input)
            loss = y.sum()
            loss.backward()
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


def thunder_fw_bw_benchmark(fw_traces: list, bw_traces: list, fw_labels: list, bw_labels: list, iters: int, nvsight: bool = False) -> None:
    assert(len(fw_traces) == len(bw_traces) == len(fw_labels) == len(bw_labels))
    for trc, label in zip(fw_traces, fw_labels):
        c, m, _ = benchmark_trace(trc, apply_del_last_used=False, snapshot=True, snapshot_name=label, iters=iters, nvsight=nvsight, nvsight_fn_name=label)
        print(f'Executing {label} trace:\n{c} ms, {m / (2**30)} GB')

    i = 0
    for trc, label in zip(bw_traces, bw_labels):
        c, m, _ = benchmark_trace(trc, apply_del_last_used=False, snapshot=True, snapshot_name=label, iters=iters, nvsight=nvsight, nvsight_fn_name=label, fw_trace=fw_traces[i])
        print(f'Executing {label} trace:\n{c} ms, {m / (2**30)} GB')
        i += 1
