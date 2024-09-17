from collections.abc import Callable
import torch
from thunder.backend_optimizer.utils import benchmark_trace
from torch.utils.benchmark import Timer, Compare

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

def _run_loss(model, input, target, loss_fn):
    logits = model(input)
    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1)
    loss = loss_fn(logits, target)
    loss.backward()

def _run_autograd(model, input):
    y = model(input)
    torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))

def torch_fw_bw_benchmark_nvsight(models: list, labels: list, inputs: list, iters: int, loss) -> None:
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


def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int, loss_fn: Callable | None = None) -> None:
    """
    Benchmark a mock trainig loop of the given models. Time measurements will be performed by using cuda events.
    A loss function is applied to trigger backward if provided. Otherwise torch.autograd will be used.
    Forward and backward pass will be recorded separately.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        iters: benchmark iterations.
    """
    for m, input, label in zip(models, inputs, labels):
        # Warm up
        target = torch.ones_like(input)
        for _ in range(warm_up_iters):
            if loss_fn is not None:
                _run_loss(m, input, target, loss_fn)
            else:
                _run_autograd(m, input)

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
            target = torch.ones_like(input)
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            y = m(input)
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            start_events[i].record(stream)
            if loss_fn is not None:
                y = y.reshape(-1, y.size(-1))
                target = target.reshape(-1)
                loss = loss_fn(y, target)
                loss.backward()
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

            max_allocated_bytes = max(max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device()))

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f"{label} tot bw time: {tot_time} ms")
        print(f"{label} max bw allocated memory: {max_allocated_bytes / (2**30)} GB")

def torch_timer_total_benchmark(
    models: list, labels: list, inputs: list, name: str = "Model", loss_fn: Callable | None = None
) -> None:
    """
    Benchmark a mock trainig loop time of the given models. Measurements will be computed by using torch.utils.benchmark.Timer.
    A loss function is applied to trigger backward if provided. Otherwise torch.autograd will be used.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        name: the model name
        loss_fn: a Pytorch loss function
    """
    results = []
    for m, l, i in zip(models, labels, inputs):
        t = Timer(
            stmt="""
            _run_loss(m, i, target, loss_fn)
            """
            if loss_fn is not None
            else """
            _run_atograd(m, i)
            """,
            globals={
                "i": i,
                "m": m,
                "target": torch.zeros_like(i),
                "_run_loss": _run_loss,
                "_run_autograd": _run_autograd,
                "loss_fn": loss_fn,
            },
            label=name,
            description=l,
        )
        results.append(t.blocked_autorange(min_run_time=1))
    compare = Compare(results)
    compare.colorize(rowwise=True)
    compare.print()

def torch_total_benchmark(models: list, labels: list, inputs: list, iters: int, loss_fn: Callable | None = None) -> None:
    """
    Benchmark a mock trainig loop of the given models. Time measurements will be performed by using cuda events.
    A loss function is applied to trigger backward if provided. Otherwise torch.autograd will be used.
    The complete time will be recorded with no split between forward pass and backward pass.

    Args:
        models: a list of Callable models to benchmark.
        labels: a list of labels (names) referring to the models.
        inputs: a list of inputs to give to models' forward pass.
        iters: benchmark iterations.
    """
    for m, input, label in zip(models, inputs, labels):
        # Warm up
        target = torch.ones_like(input)
        for _ in range(warm_up_iters):
            if loss_fn is not None:
                _run_loss(m, input, target, loss_fn)
            else:
                _run_autograd(m, input)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        torch.cuda.synchronize()
        for i in range(iters):
            target = torch.ones_like(input)
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

            start_events[i].record(stream)
            y = m(input)
            if loss_fn is not None:
                y = y.reshape(-1, y.size(-1))
                target = target.reshape(-1)
                loss = loss_fn(y, target)
                loss.backward()
            else:
                torch.autograd.grad(y, input, grad_outputs=torch.ones_like(y))
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
