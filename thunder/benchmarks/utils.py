from collections.abc import Callable
import torch
from thunder.backend_optimizer.utils import benchmark_trace

warm_up_iters = 50

class SplitFwBwBenchmarkUtils:
    """
    Represents a benchmark result container.
    It should be used when a single trace region is benchmarked as it can store an optimal executor (referred to the bsym under investigation).

    Attributes:
        cost: The benchmark result. Can be compute time or peak memory usage.
        fw_fn: Storage for a forward trace.
        bw_fn: Storage for a backward trace.
        executor: An OperatorExecutor.
    """
    def __init__(
        self, *, cost: float = float("inf"), fw_fn: Callable | None = None, bw_fn: Callable | None = None, executor=None
    ) -> None:
        self.cost: float = cost
        self.fw_fn: Callable | None = fw_fn
        self.bw_fn: Callable | None = bw_fn
        self.executor = executor


def torch_fw_bw_benchmark_nvsight(models: list, labels: list, inputs: list, iters: int) -> None:
    """
    Benchmark a mock trainig loop of the given models. The loss function is defined as a naive torch.sum().
    This util will generate nvsight system profiles.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        iters: benchmark iterations.
    """

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
            torch.cuda.nvtx.range_push("forward")
            y = m(input)
            torch.cuda.nvtx.range_pop()
            loss = y.sum()
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:
    """
    Benchmark a mock trainig loop of the given models. The loss function is defined as a naive torch.sum().
    Forward and backward pass will be both recorded.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        iters: benchmark iterations.
    """
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

            max_allocated_bytes = max(max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device()))

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f"{label} tot fw time: {tot_time} ms")
        print(f"{label} max fw allocated memory: {max_allocated_bytes / (2**30)} GB")

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

            max_allocated_bytes = max(max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device()))

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f"{label} tot bw time: {tot_time} ms")
        print(f"{label} max bw allocated memory: {max_allocated_bytes / (2**30)} GB")


def torch_total_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:
    """
    Benchmark a mock trainig loop of the given models. The loss function is defined as a naive torch.sum().
    The complete time will be recorded with no split between forward pass and backward pass.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        iters: benchmark iterations.
    """
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
            loss = y.sum()
            loss.backward()
            end_events[i].record(stream)

            max_allocated_bytes = max(max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device()))

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f"{label} tot time: {tot_time} ms")
        print(f"{label} max allocated memory: {max_allocated_bytes / (2**30)} GB")


def thunder_fw_bw_benchmark(
    fw_traces: list, bw_traces: list, fw_labels: list, bw_labels: list, iters: int, nvsight: bool = False
) -> None:
    """
    Benchmark a foward and backward trace pair.
    The requested inputs are TraceCtx objects.
    A nvsight profile can be generate if requested.

    Args:
        fw_traces: a list of TraceCtx.
        bw_traces: a list of TraceCtx.
        fw_labels: a list of labels (names) referring to the forward traces.
        bw_labels: a list of labels (names) referring to the backward traces.
        iters: benchmark iterations.
        nvsight: flag to control nvsight profile generation.
    """
    assert len(fw_traces) == len(bw_traces) == len(fw_labels) == len(bw_labels)
    for trc, label in zip(fw_traces, fw_labels):
        c, m, _ = benchmark_trace(
            trc,
            apply_del_last_used=False,
            snapshot=True,
            snapshot_name=label,
            iters=iters,
            nvsight=nvsight,
            nvsight_fn_name=label,
        )
        if not nvsight:
            print(f"Executing {label} trace:\n{c} ms, {m / (2**30)} GB")

    i = 0
    for trc, label in zip(bw_traces, bw_labels):
        c, m, _ = benchmark_trace(
            trc,
            apply_del_last_used=False,
            snapshot=True,
            snapshot_name=label,
            iters=iters,
            nvsight=nvsight,
            nvsight_fn_name=label,
            fw_trace=fw_traces[i],
        )
        if not nvsight:
            print(f"Executing {label} trace:\n{c} ms, {m / (2**30)} GB")
        i += 1
