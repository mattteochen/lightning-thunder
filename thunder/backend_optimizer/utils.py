from collections.abc import Callable, Hashable, Sequence
from typing import Any

from thunder.core.compile_data import get_compile_data
from thunder.core.dtypes import to_torch_dtype
from thunder.core.prims import PrimIDs
from thunder.core.proxies import (
    AnyProxy,
    CollectionProxy,
    FloatProxy,
    IntegerProxy,
    NumberProxy,
    Proxy,
    TensorProxy,
    Variable,
    variableify,
)
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx, from_trace, get_tracectx, reset_tracectx, set_tracectx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_always_executors
from thunder.core.utils import check, safe_map_flat
import thunder.core.transforms as transforms
from itertools import chain
import torch
from thunder.core.dtypes import dtype
from enum import Enum


class TraceType(Enum):
    """
    Represents the nature of a trace, if forward (computational) or backward.
    """

    FW = 0
    BW = 1


class BenchmarkResult:
    """
    Represents a trace benchmark result information.

    Attributes:
        time: Benchmark computation time.
        memory: Benchmark peak memory usage.
        trace: Computaiton trace.
        label: A generic label.
        index: A generic index in a sequence.
    """

    def __init__(
        self,
        *,
        time: float = float("inf"),
        memory: float = float("inf"),
        trace: TraceCtx = TraceCtx(),
        label: str | Hashable = "",
        index: int = -1,
    ) -> None:
        self.runtime: float = time
        self.memory: float = memory
        self.trace: TraceCtx = trace
        self.label: str | Hashable = label
        self.index: int = index


class OptimizerType(Enum):
    """
    Represents the autotuner target.
    """

    MEMORY = 0
    RUNTIME = 1


# Maybe we can use id(s)
def sequence_hash(s: Sequence) -> str:
    """
    Create a fake hash for a sequence of elements.
    A fake hash is created because it relies on the elements metadata and not on a specific hash function.

    Args:
        s: A sequence to hash.
    """

    def rec(s) -> str:
        name = "["
        for e in s:
            if e is None:
                name += "None#"
            elif hasattr(e, "name"):
                name += e.name + "#"
            elif isinstance(e, Sequence) and not isinstance(e, str):
                name += rec(e)
            elif isinstance(e, int):
                name += "int" + str(e) + "#"
            else:
                raise AssertionError(f"Unsupported type = {type(e)}")
        name += "]"
        return name

    return rec(s)


def can_executor_execute(ex: Executor, bsym: BoundSymbol) -> bool:
    """
    Wrap the `can_execute` call of the `Executor`.

    Args:
        ex: The executor to test.
        bsym: The bound symbol to test.
    """
    try:
        return ex.can_execute(bsym)
    except Exception:
        return False


def get_first_available_operator_executor(
    *, bsym: BoundSymbol, executors: Sequence[Executor], empty_hash: str = "empty"
):
    """
    Returns the first available executor which can execute the given bound symbol.

    Args:
        bsym: The bound symbol to execute.
        executors: A list of possible executors.
        empty_hash: A label representing an empty executor if none will be found.
    """
    for ex in executors:
        if isinstance(ex, FusionExecutor):
            continue
        if can_executor_execute(ex, bsym):
            return ex
    return Executor(name=empty_hash)


def flatten_sequence(sequence: Sequence) -> list:
    """
    Flat a sequence containing sub sequences with a dfs search.
    By default None elements will be skipped.

    Args:
        sequence: The sequence to flatten.
    """
    res = []
    for e in sequence:
        if isinstance(e, Sequence):
            res.extend(flatten_sequence(e))
        # Skip Nones as they are not useful
        elif e is not None:
            res.append(e)
    return res


def get_not_used_intermediate_outsputs(trace_in: TraceCtx) -> list[Proxy]:
    """
    Returns all the intermediate outputs that are not used or returned in the input trace.
    This can be usefull if we want to force a specific TensorProxy to be returned in a modfied trace to avoid the dce.

    Args:
        in_trace: A generic trace.
    """

    def is_in_sequence(seq: Sequence[Any], t: Proxy):
        for e in seq:
            if hasattr(e, "name") and hasattr(t, "name") and e.name == t.name:
                return True
        return False

    def unpack_output(out) -> Sequence[Proxy]:
        if issubclass(type(out), Proxy):
            return [out]
        elif isinstance(out, Sequence):
            return flatten_sequence(out)
        else:
            raise RuntimeError(f"Unpack operation not defined for {type(out)}")

    ans: list[Proxy] = []
    for a in trace_in.bound_symbols:
        f = False
        unpacked_out = unpack_output(a.output)
        for e in unpacked_out:
            # None values are checked inside the unpack_output fn
            for b in trace_in.bound_symbols:
                if b.args is not None and isinstance(b.args, Sequence) and is_in_sequence(b.args, e):
                    f = True
                    break
            if not f:
                ans.append(e)
    from thunder.backend_optimizer.optimizer import logger

    logger.debug(f"Returning not used proxies: {[p.name if hasattr(p, 'name') else p for p in ans ]}")
    return ans


def assign_executors(
    *,
    in_trace: TraceCtx,
    executors_list: list[Executor | FusionExecutor | OperatorExecutor]
    | tuple[Executor | FusionExecutor | OperatorExecutor, ...],
    always_executors: list[Executor] | tuple[Executor, ...],
    empty_str: str | Hashable,
    compile_data=None,
    fusion_executor_compile_options_to_activate: Any | None = None,
) -> TraceCtx:
    """
    Given a not optimized trace (original computation trace) generate a transformed trace with the requested executors.

    Args:
        in_trace: The computation trace.
        executors_list: A list of executors, one for each trace region. The size of this list is expected to be equal to the number of bound symbols inside the trace.
        always_executors: A list of always executors to pick up symbols not picked up by any specific executor.
        empty_str: A label representing an empty executor in the executors_list.
        compile_data: A reference to the current compilation data.
        fusion_executor_compile_options_to_activate: Any fusion exeuctor compilation options that can be enabled during the trace generation (for example nvFuser).
    """

    from thunder.executors.passes import _transform_for_operator_executor_execution

    def _assign_executors():
        swapmap: dict[Variable, Proxy] = {}

        def restore_correct_args(trace_in: TraceCtx):
            def args_eq(a, b) -> bool:
                if len(a) != len(b):
                    return False
                for obj_a, obj_b in zip(a, b):
                    if type(obj_a) == type(obj_b) and isinstance(obj_a, TensorProxy):
                        if obj_a.name != obj_b.name:
                            return False
                    elif type(obj_a) == type(obj_b) and not isinstance(obj_a, TensorProxy):
                        if obj_a != obj_b:
                            raise AssertionError(f"What do you want to do here:\nobj_a:\n{obj_a}\nobj_b:{obj_b}")
                return True

            def clear(bsym: BoundSymbol, input):
                size = len(bsym.subsymbols)
                if size > 0:
                    for subsym in bsym.subsymbols:
                        if not args_eq(subsym.args, input):
                            subsym.args = tuple(list(input))
                            clear(subsym, input)

            for bsym in trace_in.bound_symbols:
                if isinstance(bsym.sym.executor, OperatorExecutor):
                    clear(bsym, bsym.args)

        def update_swapmap(o: Any, no: Any) -> None:
            if isinstance(o, Proxy):
                check(
                    isinstance(no, Proxy),
                    lambda: f"Expected an execution transform to produce outputs with the same type, but found {type(o)} and {type(no)}",
                )

                vo = variableify(o)
                vno = variableify(no)
                if vo == vno:
                    return
                swapmap[vno] = o

        def preserve_bsym(bsym: BoundSymbol) -> Any:
            trace: TraceCtx | None = get_tracectx()
            if trace is None:
                raise AssertionError("None trace context")
            trace.scopes[-1].append(bsym)
            for p in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
                trace.names.add(p.name)
            return bsym.output

        def visit_helper(bsym: BoundSymbol, ex: Executor) -> None | bool:
            if bsym.sym.python_impl is not None:
                return None

            # We have mapped this at previous stages
            if ex.name == empty_str:
                return None

            execution_transform: None | Callable = ex.get_execution_transform(bsym.sym)
            out: Any
            if execution_transform is not None:
                out = execution_transform(*bsym.args, **bsym.kwargs)
            elif isinstance(ex, OperatorExecutor):
                # Calls the operator executor's operation
                op: Symbol | None = ex.implmap[bsym.sym.id].symbol
                if op is None:
                    raise AssertionError("op is None")
                out = op(*bsym.args, **bsym.kwargs)
            elif isinstance(ex, FusionExecutor):
                # Preserves the symbol as is (it will be handled in the fusion pass)
                out = preserve_bsym(bsym)
            else:
                raise AssertionError("Unknown executor")

            safe_map_flat(update_swapmap, bsym.output, out)

            return True

        def visit(bsym: BoundSymbol, ex: Executor) -> transforms.VISIT_TYPE:
            return transforms.VISIT_TYPE.NO_OP if visit_helper(bsym, ex) is None else transforms.VISIT_TYPE.REPLACE

        if len(executors_list) != len(in_trace.bound_symbols):
            raise AssertionError("len(executors_list) != len(in_trace.bound_symbols)")

        cached_subsymbols: dict[str, Sequence[BoundSymbol]] = {}
        executor_mapping: dict[str, Executor] = {}
        unique_fusion_executors = set()

        # Input should have equal length
        if len(executors_list) != len(in_trace.bound_symbols):
            raise AssertionError("len(executors_list) != len(extrace.bound_symbols)")

        for b, e in zip(in_trace.bound_symbols, executors_list):
            if isinstance(e, FusionExecutor):
                unique_fusion_executors.add(e)
            if isinstance(b.output, TensorProxy):
                executor_mapping[b.output.name] = e

        extrace = transforms.visitor_transform_paired(in_trace, visit, zip(in_trace.bound_symbols, executors_list))

        # Restores original variables
        bound_symbols: list[BoundSymbol] = []
        for bsym in extrace.bound_symbols:
            nbsym: BoundSymbol = bsym.from_bsym_swap_proxies(swapmap)
            bound_symbols.append(nbsym)
        extrace.bound_symbols = bound_symbols

        for bsym in extrace.bound_symbols:
            if isinstance(bsym.output, TensorProxy):
                t_name = bsym.output.name
                if t_name not in executor_mapping:
                    # Symbol added by the visitor
                    continue
                saved_ex = executor_mapping[t_name]
                if isinstance(saved_ex, OperatorExecutor):
                    cached_subsymbols[t_name] = list(bsym.subsymbols)
                    # This will leave out these symbols from the fusion pass
                    bsym.subsymbols = []

        # Perform fusion pass
        for ex in unique_fusion_executors:
            extrace = ex.fusion_pass(extrace)

        # Restore subsymbols
        # TODO (matteochen): Improve this search
        for k, v in cached_subsymbols.items():
            # NOTE: Some symbols may be cut out by the fusion pass -> CSE
            # For example:
            # a = 1 + 1
            # b = 1 + 1
            # c = a + b
            # being replaced by c = a + a
            for bsym in extrace.bound_symbols:
                if isinstance(bsym.output, TensorProxy) and bsym.output.name == k:
                    bsym.subsymbols = v

        restore_correct_args(extrace)

        # Apply always executors
        extrace = _transform_for_operator_executor_execution(extrace, always_executors)

        return extrace

    if fusion_executor_compile_options_to_activate:
        return wrap_fn_with_exeuctor_compile_option(fusion_executor_compile_options_to_activate, _assign_executors)
    return _assign_executors()


def operation_in_trace(*, trace: TraceCtx, op: str, prefix: bool = False) -> bool:
    """
    Test if an operation is being used inside a trace.

    Args:
        trace: A computation trace.
        op: The operation name to be tested.
        prefix: Test only the prefix label.
    """

    # This is to query nv_enable_bookend (https://github.com/Lightning-AI/lightning-thunder/blob/339a782e3d75061a065a3d2e47b5206f23aea7c3/thunder/executors/nvfuserex_impl.py#L807)
    # as there won't be any references about this in a trace.
    always_true = set(["bookend"])

    if op in always_true:
        return True
    for b in trace.bound_symbols:
        if prefix:
            if b.sym.name.startswith(op):
                return True
        else:
            if b.sym.name == op:
                return True
    return False


def is_te_used(trace: TraceCtx) -> bool:
    """
    Test if transformer engine is being used inside a trace.

    Args:
        trace: A computation trace.
    """
    from thunder.executors.transformer_engineex import linear_bound_symbol_name_prefix
    from thunder.executors.transformer_engineex import te_functional_linear_backward_name

    if operation_in_trace(trace=trace, op=te_functional_linear_backward_name) or operation_in_trace(
        trace=trace, op=linear_bound_symbol_name_prefix, prefix=True
    ):
        return True

    return False


def is_backward_trace(trace: TraceCtx) -> bool:
    """
    Test if a trace is a backward trace from its signature.

    Args:
        trace: A computation trace.
    """
    sig = trace.signature_with_no_ctx()
    return sig.find("backward") >= 0


def benchmark_trace(
    trace: TraceCtx,
    iters: int = 1,
    show_func=False,
    apply_del_last_used=True,
    snapshot=False,
    snapshot_name="",
    nsight: bool = False,
    nsight_fn_name: str = "",
    **kwargs,
) -> tuple[float, float, Any]:
    """
    Benchmark a generic computation trace compute time and peak memory usage.
    nsight profiles can be generated if requested.

    If a backward trace is benchmarked, its paired forward trace is requested (with kwargs) as we don't generate inputs
    for the backward call from the static args but with the dynamic arguments returned by the forward trace.

    Args:
        trace: A computation trace.
        iters: Benchmark iterations.
        show_func: Print the executed trace if True.
        apply_del_last_used: A flag to control if the trace should be executed after a deletion of not used vars call.
        snapshot: A flag controlling if memory usage snapshots should be created (https://pytorch.org/docs/stable/torch_cuda_memory.html).
        snapshot_name: A label for the generated snapshot.
        nsight: A flag contolling if nvsigh profiles should be generated or not.
        nsight_fn_name: A label for the nsight iteration name during benchmark loop.
    """
    from thunder.executors.passes import del_last_used
    import inspect

    torch.compiler.reset()

    # TODO: If TE is used inside the trace we have to clone the input arguments as
    # we are currently seeing benchmarking issues at the iteration i > 0
    def clone_args_if_needed(args):
        te_used = is_te_used(trace)
        if not te_used:
            return args
        res = []
        # Detatching the tensors as for standalone trace benchmarks we are not interested in the gradients
        for arg in args:
            if isinstance(arg, Sequence):
                res.append(clone_args_if_needed(arg))
            else:
                if isinstance(arg, torch.Tensor):
                    res.append(arg.clone().detach())
                else:
                    res.append(arg)
        return tuple(res)

    def compute_time_cost_nsight(fn: Callable, iters: int, *args) -> tuple[float, float, Any]:
        try:
            warm_up_iters = 50
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Warm up cycles
            for _ in range(warm_up_iters):
                new_args = clone_args_if_needed(args)
                fn(*new_args)
            # Benchmark
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(iters):
                new_args = clone_args_if_needed(args)
                torch.cuda.empty_cache()
                torch.cuda.nvtx.range_push(f"thunder benchmark fn:{nsight_fn_name}, iter{i}")
                fn(*new_args)
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

            return float("inf"), float("inf"), None
        except Exception as e:
            import inspect

            trc = inspect.getsource(fn)
            print(f"#Trace execution failed for nsight (error: {e}):\n{trc}")
            raise e

    def compute_time_cost_ms(fn: Callable, repr: str, iters: int, *args) -> tuple[float, float, Any]:
        try:
            current_iter = 0
            warm_up_iters = 50
            out = None

            # Warm up cycles
            for _ in range(warm_up_iters):
                new_args = clone_args_if_needed(args)
                out = fn(*new_args)
            # Snapshot request
            if snapshot:
                new_args = clone_args_if_needed(args)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.memory._record_memory_history()
                fn(*new_args)
                torch.cuda.memory._dump_snapshot(snapshot_name + "_benchmark.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
            # Benchmark
            stream = torch.cuda.current_stream()
            max_allocated_bytes = 0
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            torch.cuda.synchronize()
            for i in range(iters):
                current_iter = i
                new_args = clone_args_if_needed(args)
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                torch.cuda.empty_cache()
                torch.cuda._sleep(1_000_000)
                start_events[i].record(stream)
                fn(*new_args)
                end_events[i].record(stream)
                max_allocated_bytes = max(
                    max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device())
                )

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            tot_time = sum(times) / iters
            return tot_time, max_allocated_bytes, out
        except Exception as e:
            print(f"#Trace execution failed at iter {current_iter} (error: {e})\n{repr}")
            raise e

    def build_static_args(sequence: Sequence, **kwargs) -> list:
        return transform_proxies_to_real(sequence, level=0, **kwargs)

    def backward_trace_args_preprocess() -> list:
        if "fw_trace" not in kwargs:
            raise RuntimeError(
                "Set the associated forward trace in order to benchmark backward pass with sdpa executor"
            )
        fw_trace = kwargs.get("fw_trace", None)
        if not isinstance(fw_trace, TraceCtx):
            raise AssertionError(f"forward trace is not a TraceCtx. Received: {type(fw_trace)}")
        # Run the fw trace and get the outputs
        fw_output = benchmark_trace(fw_trace, apply_del_last_used=False)[2]

        # Check if the fw trace is a final trace or an intermediate one (used for single trace region benchmarks)
        sig = fw_trace.signature_with_no_ctx()
        is_fw_final_trace = sig.startswith("def augmented")

        # Filter the C0 tuple
        # These location might change if the implementation of the automatic
        # differentiation transform changes. The saved tensors are the second output
        # of the return statement. There's a prototype changing the saved tensors to
        # be part of the output of a special symbol
        # https://github.com/Lightning-AI/lightning-thunder/pull/214
        saved_for_bw_C0 = fw_output[1] if not is_fw_final_trace else fw_output[1][0]

        # The underlying API will generate TE.Float8 tensors also, hence it must know if TE executor is used or not
        input_args = build_static_args(trace.args, te_used=te_used)

        # Now, we expected that if the fw trace is a final trace also the bw trace is a final one. And vice versa
        if is_fw_final_trace:
            # Swap saved_for_backward_traces
            saved_for_bw = saved_for_bw_C0, fw_output[1][1]  # Saved for backward tuple unpacks in (C0, _)
            # Subsitute the static inputs for saved_for_backward with the runtime ones
            input_args.pop(0)
            input_args.insert(0, saved_for_bw)
        else:
            # Currently single trace region backward trace receives as input the saved_for_bw tensors plus some others.
            # They are indexed like [saved_for_bw, others...].
            # NOTE: This may change in the future.
            """
            Example:
                @torch.no_grad()
                @no_autocast
                def _cudnn_sdpa_bwd_wrapper(query, key, value, attn_mask, dropout_p=0.0, is_causal=False, *, scale=None):
                  # query: "cuda:0 bf16[32, 8, 128, 64]"
                  # key: "cuda:0 bf16[32, 8, 128, 64]"
                  # value: "cuda:0 bf16[32, 8, 128, 64]"
                  # dropout_p: "float 0.0"
                  # is_causal: "bool False"
                  (t0, t1, t2, t3) = cudnn_sdpa_fwd(query, key, value, None, dropout_p, is_causal, scale=None)
                  return (t0, [query, key, value, dropout_p, is_causal, t0, t1, t2, t3])

                @torch.no_grad()
                @no_autocast
                def scaled_dot_product_attention_backward(query, key, value, dropout_p, is_causal, t0, t1, t2, t3, t4):
                  (t5, t6, t7) = cudnn_sdpa_bwd(t4, query, key, value, None, dropout_p, is_causal, t0, t1, t2, t3, scale=None, cat_grad_qkv=False)
                  return {'query': t5, 'key': t6, 'value': t7, 'attn_mask': None, 'dropout_p': None, 'is_causal': None, 'scale': None}

            See how the backward trace needs t4 as argument recoveered from the static args
            """
            updated_input_args = [t for t in saved_for_bw_C0]
            updated_input_args.extend(
                input_args[len(updated_input_args) :]
            )  # Should be only one variable but leave this dyanamic
            input_args = updated_input_args

        return input_args

    # Check for correctness
    if trace.bound_symbols[-1].sym.id != PrimIDs.RETURN:
        raise AssertionError("Missing return statement")

    if apply_del_last_used:
        trace = del_last_used(trace)

    # Handle TE traces
    cd = get_compile_data()
    # We might benchmarking a partial trace where the TE symbol is not included yet, in this case rely on the compile option which tells us
    # that afterwards at least one TE symbol will be included
    # NOTE: compile data could be None if this benchmark util is used outside the compilation process. If this is the case we are benchmarking
    # a whole trace (in theory) and is_te_used API will return the needed result.
    te_used = (cd.compile_options.get("te_used", False) if cd else False) or is_te_used(trace)
    if te_used:
        cached_te_fp8_autocast_value = trace._include_te_fp8_autocast
        trace._include_te_fp8_autocast = True

    # Build trace arguments: forward trace will receive compile time tensors while
    # backward trace will receive dynamic inputs (runtime) to match real training env.
    if is_backward_trace(trace):
        input_args = backward_trace_args_preprocess()
    # Forward or computational trace, parse the compile time input args...
    else:
        input_args: list = build_static_args(trace.args, te_used=te_used)

    # Obtain the python executable string
    executable_str = trace.python()
    executable = trace.python_callable()
    if show_func:
        print(inspect.getsource(executable))

    trace_tok = set_tracectx(trace)

    t = float("inf")
    m = float("inf")
    answer = None
    try:
        if nsight:
            t, m, answer = compute_time_cost_nsight(executable, iters, *input_args)
        else:
            t, m, answer = compute_time_cost_ms(executable, executable_str, iters, *input_args)
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        reset_tracectx(trace_tok)

    # Restore the autocast value to not mess up the input trace
    if te_used:
        trace._include_te_fp8_autocast = cached_te_fp8_autocast_value

    return t, m, answer


def _register_impl_executor(ex: Executor, id: PrimIDs, fn: Callable, checker: Callable) -> None:
    if ex.name == "nvfuser":
        from thunder.executors.nvfuserex_impl import register_supported

        register_supported(id, fn, checker)


def _recover_ex_from_compile_option(option: str) -> Executor:
    if option.startswith("nv"):
        from thunder.executors.nvfuserex_impl import ex

        return ex
    else:
        raise AssertionError(f"Compile option not recognized: {option}")


def wrap_fn_with_exeuctor_compile_option(option, fn: Callable | None = None, *args):
    """
    Wraps a function call enabling a compile option for a specific executor.
    The compile option will be restored after the function completes.
    This can be usefull if we want to benchmark a specific compile option.

    Args:
        option: The option to be enabled.
        fn: A callable function.
        args: Function arguments.
    """
    from thunder.core import compile_data

    cd = compile_data.get_compile_data()
    if option is not None:
        # Update compile option context
        if cd is None:
            raise AssertionError("compile_data is None")
        old_opt: bool | None = cd.compile_options.get(option.fusion_tag, None)
        new_opt = True if old_opt is None or old_opt is False else False
        cd.compile_options[option.fusion_tag] = new_opt
        # Register the impl for the executor in order to be able to execute the id
        _register_impl_executor(
            _recover_ex_from_compile_option(option.fusion_tag),
            option.id,
            option.impl,
            option.checker,
        )
    # Call fn and return output
    if fn:
        out = fn(*args)
    else:
        out = None
    # Restore compile option
    if option is not None:
        cd.compile_options[option.fusion_tag] = old_opt

    return out


def print_trace_args(trace: TraceCtx):
    """
    Utility to display a trace arguments.

    Args:
        trace: A computation trace.
    """
    print_nested_sequence(trace.args)


def print_nested_sequence(args, show_dicts=False):
    """
    Utility to display a sequence of elements with possible nested sequences.
    Elements will be retrieved in a dfs manner.

    Args:
        args: The input sequence.
        show_dicts: Control if dict types should be printed.
    """

    import pprint

    def is_tensor(t):
        return isinstance(t, torch.Tensor) or isinstance(t, TensorProxy)

    if not isinstance(args, Sequence):
        return
    print("###################################### Sequence start")

    def _print(args, level):
        tabs = "\t" * level
        print(f"Level {level} start")
        for arg in args:
            if isinstance(arg, Sequence):
                _print(arg, level + 1)
            else:
                tensor_shape = arg.shape if is_tensor(arg) else None
                dtype = arg.dtype if is_tensor(arg) else None
                name = arg.name if isinstance(arg, TensorProxy) else ""
                print(
                    f'{tabs}{name + ": " if name else ""}{type(arg)}{pprint.pformat(arg) if isinstance(arg, dict) and show_dicts else ""} {tensor_shape if tensor_shape else ""} {dtype if dtype else ""}'
                )
        print(f"Level {level} end")

    _print(args, 0)
    print("###################################### Debug args\n")


def update_compile_options_executor_list_after_fw_bw_split() -> None:
    """
    Updates the compile options with the executors that have been placed by the forward-backward split pass.
    This utility can be used to save all the executors that have been effectively placed in a trace.
    """

    cd = get_compile_data()
    assert cd

    # Get all the possible options that the vjp_optimization pass will use
    options: dict = get_fw_bw_split_backends_options(
        autotune_enable_te=cd.compile_options.get("autotune_enable_te", False)
    )
    executors_list = list(cd.executors_list)

    # Remove all the initial options
    for _, v in options.items():
        for ex in v:
            if ex in executors_list:
                executors_list.remove(ex)

    # Putting at the front even though order does not matter
    for ex in cd.compile_options["autotune_executors_placed_by_fw_bw_split"]:
        executors_list.insert(0, ex)

    # Assign new compilation executors options
    cd.executors_list = executors_list


def transform_tensor(arg: TensorProxy, **kwargs) -> torch.Tensor:
    """
    Retrive the associated torch.Tensor from a proxy tensor by reading its metadata.
    This will allocate the real tensor in memory.
    This utility can read transformer engine compilation requests and generate the associated FP8 tensor if needed.

    Args:
        arg: The proxy tensor.
    """
    from thunder.core.dtypes import is_float_dtype, is_signedinteger_dtype, is_boolean_dtype

    dtype = arg.dtype
    shape = arg.shape
    device = arg.device
    requires_grad = arg.requires_grad
    torch_dtype = to_torch_dtype(dtype)
    if torch_dtype is None:
        raise AssertionError(f"Unrecognized thunder dtype: {dtype}")
    if is_float_dtype(dtype):
        # Use TE Float8 if TE is enabled, it has float32 torch dtype
        te_used = kwargs.get("te_used", False)
        if te_used:
            tensor: torch.Tensor = torch.randn(
                shape,
                dtype=torch_dtype if dtype.bytes > 1 else torch.float32,
                device=device.device_str(),
                requires_grad=requires_grad,
            )
            if dtype.bytes == 1:
                import transformer_engine.pytorch as te

                tensor = te.float8_tensor.Float8Tensor.to_float8(tensor)
        # Support standard float tensors
        else:
            tensor: torch.Tensor = torch.randn(
                shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad
            )
    elif is_signedinteger_dtype(dtype):
        tensor: torch.Tensor = torch.randint(
            0, 8, shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad
        )
    elif is_boolean_dtype(dtype):
        # TODO (matteochen): maybe random?
        tensor: torch.Tensor = torch.zeros(
            *shape, dtype=torch.bool, device=device.device_str(), requires_grad=requires_grad
        )
    else:
        raise AssertionError(f"dtype {dtype} not supported yet")

    return tensor


def transform_proxies_to_real(sequence: Sequence, level=0, **kwargs) -> tuple | list:
    """
    Retrieve a sequence of real arguments relative to a sequence of proxy arguments.
    This supports also nested sequences in a recursive way.

    Args:
        sequence: The input proxy sequence.
        level: An utility integer representing the search dept.
    """
    from thunder.executors.transformer_engineex import Context as C

    res = []
    for e in sequence:
        if type(e) is tuple:
            res.append(transform_proxies_to_real(e, level + 1, **kwargs))
        else:
            if isinstance(e, TensorProxy):
                res.append(transform_tensor(e, **kwargs))
            elif isinstance(e, IntegerProxy):
                if e.python_type is bool:
                    res.append(False if e.value is None else e.value)
                else:
                    res.append(0 if e.value is None else e.value)
            elif isinstance(e, FloatProxy):
                res.append(0.0 if e.value is None else e.value)
            # Transformer engine Context object
            #
            # This instruction will populate the args with a dummy context which is not correct in theory.
            # For the benchmark purpose (where this fn is currently used) this error will not impact on the runtime correctness as at the end we
            # will use the cached runtime contexts from the forward pass.
            # We need this only to generate a context for the static inputs (which are discarded afterwards).
            #
            # Backward args: (saved_for_backward, cotangents)
            # saved_for_backward -> replaced by the runtime tuple
            # cotangents -> static inputs will be used
            # If the static input generator will be capable to generate only the cotangents then branch will not be used anymore
            #
            # Currently an option to fill a custom maybe real context is left.
            elif hasattr(e, "name") and isinstance(e, AnyProxy) and e.name.startswith("ctx_te"):
                required_context = kwargs.get("cached_fw_te_ctx_out", None)
                res.append(required_context if required_context is not None else C())
            elif e is None:
                res.append(None)
            else:
                raise AssertionError(
                    f'Input arg type not recognized: {type(e)} with name: {e.name if hasattr(e, "name") else "unknown"} with value: {e}'
                )
    # Outer container must be a list
    return tuple(res) if level > 0 else res


def reorder_executors_list(executors: Sequence, **kwargs):
    """
    Reorders a random executors list to be compatible with the autotuner compilation flow.
    This will put in the front of the returned list all the executors with a grad fn.
    All the other executors will be appended afterwards.

    If no fusion executors is present inside the input list, a default one will be added in order to trigger the autotuning process.

    Args:
        executors: The executors to be reordered.
    """
    from thunder.executors.torch_compile import torch_compile_ex
    from thunder.executors.nvfuserex_impl import ex as nvfuser_ex

    reordered = []
    options = get_fw_bw_split_backends_options(**kwargs)

    are_inputs_names = isinstance(executors[0], str)

    # Put these in front to be picked up by _get_gradfn_and_executor
    for _, v in options.items():
        for ex in v:
            if are_inputs_names:
                if ex.name in executors:
                    reordered.append(ex.name)
            elif ex in executors:
                reordered.append(ex)

    # Add others
    for ex in executors:
        if ex not in reordered:
            reordered.append(ex)

    # NOTE: Currently the autotuner expects at least one Fusion executor otherwise it won't work.
    # If other techniques will be added then this constraint will not be necessary
    found = False
    for ex in reordered:
        if are_inputs_names and (ex == nvfuser_ex.name or ex == torch_compile_ex.name):
            found = True
        elif ex == nvfuser_ex or ex == torch_compile_ex:
            found = True
    if not found:
        reordered.insert(0, nvfuser_ex.name if are_inputs_names else nvfuser_ex)

    return reordered


def symbol_hash(
    *, bsym: BoundSymbol, ignore_returns_meta: bool = False, ignore_unpacks_meta: bool = False, ignore_unpacks: bool = False
):
    """
    Hash a bound symbol relying on its metadata (symbol name, bound symbol inputsa and outputs).
    No hash functions will be applied in order to leave the output readable.

    Args:
        bsym: A bound symbol.
        ignore_returns_meta: If True, return statement metadata will be ignored
        ignore_unpacks_meta: If True, unpack statements metadata will be ignored
        ignore_unpacks: If True, unpack symbols will not be included.
    """

    def _tensor_hash(t: TensorProxy) -> str:
        assert t.dtype
        shapes = [str(s) for s in t.shape]
        return "{" + "-".join(shapes) + "," + str(t.device) + "," + t.dtype.full_name + "," + str(t.requires_grad) + "}"

    def _collection_hash(c: CollectionProxy) -> str:
        return "{Collection," + c.name + "," + str((type(c.collection()))) + "," + str(len(c.collection())) + "}"

    def _number_hash(t: NumberProxy) -> str:
        return "{" + str(t.value) + "}"

    def _any_proxy_hash(p: AnyProxy) -> str:
        return "{" + p.__repr__() + "}"

    def _sequence_hash(s: Sequence | None) -> str:
        if s is None:
            return "None"

        ret = "["
        for e in s:
            if e is None:
                ret += "{None},"
            elif isinstance(e, TensorProxy):
                ret += _tensor_hash(e) + ","
            elif isinstance(e, NumberProxy):
                ret += _number_hash(e) + ","
            elif isinstance(e, Sequence):
                ret += _sequence_hash(e) + ","
            elif isinstance(e, AnyProxy):
                ret += _any_proxy_hash(e) + ","
            elif isinstance(e, CollectionProxy):
                ret += _collection_hash(e) + ","
            elif isinstance(e, int) or isinstance(e, float) or isinstance(e, bool):
                ret += "{" + f"{(type(e))}" + "},"
            elif isinstance(e, dtype):
                ret += "{" + f"{(type(e))}" + "},"
            else:
                raise RuntimeError(f"Not implemented {type(e)}. Failed bsym: {bsym}")
        return ret + "]"

    def _hash(bsym: BoundSymbol) -> str:
        match = {
            TensorProxy: _tensor_hash,
            tuple: _sequence_hash,
            list: _sequence_hash,
            Sequence: _sequence_hash,
            CollectionProxy: _collection_hash
        }

        if ignore_returns_meta and bsym.sym.id == PrimIDs.RETURN:
            return '{return}'

        if ignore_unpacks and bsym.sym.name.startswith('unpack'):
            return ''
        elif ignore_unpacks_meta and bsym.sym.name.startswith('unpack'):
            if bsym is not None and bsym.output is not None:
                if isinstance(bsym.output, Sequence) and len(bsym.output) < 1:
                    return ''
            return '{general_unpack}'

        h = bsym.sym.name
        # Handle tensor as output or sequences
        if type(bsym.output) not in match.keys():
            raise RuntimeError(f"type {type(bsym.output)} not implemented")
        h += (
            "#out:"
            + match[type(bsym.output)](bsym.output)
            + "#in:"
            # Args is always a tuple
            + _sequence_hash(bsym.args)
        )
        return h

    h = _hash(bsym)
    return ("{" + h + "}") if h else h


# Both lhs and rhs are included in the range
# TODO: known_points can be used to detect start and end of a block sequence
def repetead_trace_blocks(
    *, trace: TraceCtx, min_block_size=2, known_points: tuple[BoundSymbol, BoundSymbol] | None = None
) -> list[tuple[int, int]]:
    """
    Detects if are there repeated sections inside a given trace.
    This utility can be employed on traces referring to transformer based models where the layers are repeated N times.

    The return list will contain a tuple of two elements pointing to the index (in the computation trace) of where a block starts and ends (both included).

    The variable min_block_size can be tuned in order to not allucinate this function by capturing unwanted sections (small sections) if no repeated transformer layers can be found.

    Args:
        trace: A computation trace.
        min_block_size: The minimum block lenght, by default 2.
        known_points: If a practitioner already knows where a transformer layer starts and ends inside a given trace, these points can be supplied in order to speed up the search. Currently not implemented.
    """
    if min_block_size < 2:
        return []

    if known_points is not None:
        raise RuntimeError("known_points research is not supported.")

    symbols = [
        s
        for s in trace.bound_symbols
        if not s.sym.name.startswith("python_del") and not s.sym.name.startswith("unpack")
    ]

    def _tuple_name(tup: Sequence):
        ret = "("
        for e in tup:
            if e is None:
                ret += "None, "
            elif hasattr(e, "name"):
                ret += e.name + ", "
            elif isinstance(e, Sequence):
                ret += _tuple_name(e) + ", "
            elif isinstance(e, int) or isinstance(e, float) or isinstance(e, bool):
                ret += str(e) + ", "
            else:
                raise RuntimeError(f"Not implemented {type(e)}")
        return ret + ")"

    # Only bsym that have inputs and outputs
    original_map_indexes = {
        str(bsym.output.name) if isinstance(bsym.output, TensorProxy) else _tuple_name(bsym.output): i
        for i, bsym in enumerate(trace.bound_symbols)
        if not (bsym.output is None or not bsym.args) and bsym.sym.id != PrimIDs.RETURN
    }

    def _lcs(start_indexes) -> int:
        max_last_len = len(symbols) - 1
        max_first_len = start_indexes[1]

        lcs = 0
        while start_indexes[0] < max_first_len and start_indexes[-1] < max_last_len:
            # Get all the hashes
            hashes = [symbol_hash(bsym=symbols[i]) for i in start_indexes]
            # Advance if all the hashes coincides
            uniques = set(hashes)
            if len(uniques) == 1:
                start_indexes = [i + 1 for i in start_indexes]
                lcs += 1
            else:
                return lcs
        return max(lcs, 1)

    def _skip(bsym: BoundSymbol) -> bool:
        return bsym.output is None or not bsym.args

    bsym_indexes: dict[str, list[int]] = {}
    for i, bsym in enumerate(symbols):
        if i == len(symbols) - 1:
            break
        if _skip(bsym):
            continue
        h = symbol_hash(bsym=bsym)
        if h in bsym_indexes:
            bsym_indexes[h].append(i)
        else:
            bsym_indexes[h] = [i]

    def _range_seen(index: int, s: set):
        for r in s:
            if index >= r[0] and index <= r[1]:
                return True
        return False

    seen_hashes = set()
    seen_ranges = set()
    max_lcs = 0
    res = []
    for i, bsym in enumerate(symbols):
        if i == len(symbols) - 1:
            break
        if _skip(bsym):
            continue

        h = symbol_hash(bsym=bsym)
        # Normally, bsym are expected to output a TensorProxy
        if not isinstance(bsym.output, Proxy) or h in seen_hashes or _range_seen(i, seen_ranges):
            continue

        indexes = bsym_indexes.get(h, [])
        seen_hashes.add(h)
        if len(indexes) < 2:
            continue

        # Now we can find the longest common sequence between all the occurences
        lcs = _lcs(indexes)
        # print('\n####################')
        # for index in indexes:
        #     print(f'For index {index} lcs: {lcs}')
        #     print(f'Starting bsym: {symbols[index]}')
        #     print(f'Ending bsym: {symbols[index + lcs - 1]}')
        # print('\n####################')
        if lcs > 1:
            # Push every seen ranges to ignore all the subranges
            for i in indexes:
                seen_ranges.add((i, i + lcs - 1))

            # Set result
            if lcs > max_lcs:
                max_lcs = lcs
                res = [(i, i + lcs - 1) for i in indexes]

    if max_lcs < min_block_size:
        return []

    from thunder.backend_optimizer.optimizer import logger
    logger.debug(f"Max block lcs fouund: {max_lcs}")
    logger.debug(f"{[(symbols[r[0]].output.name, symbols[r[1]].output.name) for r in res]}")

    return [
        (original_map_indexes[symbols[t[0]].output.name], original_map_indexes[symbols[t[1]].output.name]) for t in res
    ]


def _regions_between_blocks(trace: TraceCtx, common_blocks: list[tuple]) -> int:
    """
    Retrieve the size of a gap region between common blocks.

    What is regions_between_blocks?
    They are trace regions between one transformer block and the next one (usually found in the backward trace) and given that these regions are not present
    at the end of the last transformer block it means that they are needed in order to prepare shapes or strides
    for the block at i+1 from the output of block i.
    For example if common blocks looks like: [(32, 155), (157, 280)]
    the symbol at index 156 (the gap) could generally be: <t939 = ltorch.reshape(t938, -1, 4096)  # t939: "cuda:0 f32[128, 4096]"> (for torch.float32 dtype, if another dtype is used the trace may contain other ops in this region leading to a larger gap).
    In the forward trace we have not these gaps (so far).

    In the example above the returned value will be 1.

    Args:
        trace: A computation trace.
        common_blocks: A list containig the common blocks for the given trace.

    """

    def _assert_args(seq_a: Sequence, seq_b: Sequence):
        assert len(seq_a) == len(seq_b)
        for a, b in zip(seq_a, seq_b):
            assert type(a) == type(b)
            if isinstance(a, TensorProxy):
                assert a.shape == b.shape
                assert a.dtype == b.dtype
            elif isinstance(a, Sequence):
                _assert_args(a, b)

    regions_between_blocks = common_blocks[1][0] - common_blocks[0][1] - 1
    trace_region_between_common_blocks = trace.bound_symbols[common_blocks[0][1] + 1 : common_blocks[1][0]]
    for i in range(1, len(common_blocks)):
        if not common_blocks[i][0] - common_blocks[i - 1][1] - 1 == regions_between_blocks:
            raise AssertionError(
                "Trace configuration not supported. All the trace regions between common blocks are expected to have the same number of instructions."
            )

        # Check that the trace regions are equal
        test_trace_regions = trace.bound_symbols[common_blocks[i - 1][1] + 1 : common_blocks[i][0]]
        assert len(test_trace_regions) == len(trace_region_between_common_blocks)
        for a, b in zip(test_trace_regions, trace_region_between_common_blocks):
            assert a.sym.name == b.sym.name
            _assert_args(a.args, b.args)

    return regions_between_blocks


def _indices_to_exclude_between_common_blocks(common_blocks: list[tuple]) -> list:
    """
    Retrive the indicies referring to the gaps between one common block and the next one.

    Args:
        common_blocks: A computed common block list for a given trace.
    """
    if len(common_blocks) < 2:
        return []

    ret = []
    for i in range(1, len(common_blocks)):
        start_gap_index = common_blocks[i - 1][1] + 1
        end_gap_index = common_blocks[i][0] - 1
        ret.extend([j for j in range(start_gap_index, end_gap_index + 1)])
    return ret


def reduce_common_trace_blocks(
    *, trace: TraceCtx, common_blocks_in: list[tuple], skip_between_blocks: bool = True
) -> TraceCtx:
    """
    Generate a reduced trace (shorter computation nodes) given a common block pattern.

    This can be useful to speed up the executor tuning for models with repeated layers.

    Args:
        trace: A computation trace.
        common_blocks_in: A previously computed common block pattern.
        skip_between_blocks: A flag to control if gaps between common blocks should be included in the output trace or not. See _regions_between_blocks.
    """

    def _exclude(blocks: list[tuple[int, int]], index: int, black_list: set):
        # Exclude if the index is in a repeated block
        for block in blocks:
            if index >= block[0] and index <= block[1]:
                return True

        # Exclude if it marked as to remove
        if index in black_list and skip_between_blocks:
            return True
        return False

    def _find_bsym_index(out_name: str, space: Sequence[BoundSymbol]) -> int:
        for i, b in enumerate(space):
            if b.output is not None and hasattr(b.output, "name") and b.output.name == out_name:
                return i
        raise RuntimeError(f"Can not found bsym with output {out_name} in the search space.")

    common_blocks = list(common_blocks_in)
    if len(common_blocks) < 2:
        trc = from_trace(trace)
        trc.bound_symbols = list(trace.bound_symbols)
        return trc

    # Create a mapping where we can easily find to which block a specific output belongs
    output_to_block: dict[str, tuple[int, int]] = {}
    for n_block, block in enumerate(common_blocks):
        for i in range(block[0], block[1] + 1):
            bsym = trace.bound_symbols[i]
            if not hasattr(bsym.output, "name"):
                continue
            output_to_block[bsym.output.name] = (n_block, i - block[0])

    # Check that we maintain the pattern
    regions_between_blocks = _regions_between_blocks(trace, common_blocks)

    # We have to exlude these gaps indices from the reduce trace
    index_gaps_to_exclude = []
    if regions_between_blocks:
        index_gaps_to_exclude = _indices_to_exclude_between_common_blocks(common_blocks)
    # Make it fast to search in
    index_gaps_to_exclude = set(index_gaps_to_exclude)

    # Create reduced trace regions
    bound_symbols: list[BoundSymbol] = [
        b for i, b in enumerate(trace.bound_symbols) if not _exclude(common_blocks[1:], i, index_gaps_to_exclude)
    ]

    # Now, we have to update the trace region inputs after the last block to accepts the outputs of the first block, if it's not the return statement.
    if trace.bound_symbols[common_blocks[-1][1] + 1].sym.id != PrimIDs.RETURN:
        symbol_to_correct_index = _find_bsym_index(
            trace.bound_symbols[common_blocks[-1][1] + 1].output.name, bound_symbols
        )
        symbol_to_correct = bound_symbols[symbol_to_correct_index]

        def _correct_args(target: BoundSymbol):
            args = []
            for arg in target.args:
                if arg is None:
                    args.append(None)
                elif hasattr(arg, "name") and arg.name in output_to_block:
                    _, index_in_block = output_to_block[arg.name]
                    # Recover the argument from the first block
                    args.append(trace.bound_symbols[common_blocks[0][0] + index_in_block].output)
                elif isinstance(arg, Sequence):
                    raise RuntimeError("Not implemented")
                else:
                    args.append(arg)
            return args

        def _correct_bsym(bsym: BoundSymbol) -> BoundSymbol:
            bsym = bsym.from_bsym(args=_correct_args(bsym))
            return bsym

        new_subsymbols = []
        for sub in symbol_to_correct.subsymbols:
            new_sub = _correct_bsym(sub)
            new_subsymbols.append(new_sub)

        bound_symbols[symbol_to_correct_index] = symbol_to_correct.from_bsym(
            args=_correct_args(symbol_to_correct), subsymbols=new_subsymbols
        )

    # We need to check also the return statements as we have fewer args now
    flatten_bsyms = flatten_sequence([b.output for b in bound_symbols])
    args_remained = set([b.name for b in flatten_bsyms if b is not None and hasattr(b, "name")])
    # Fw trace
    if isinstance(bound_symbols[-1].args[0], dict):
        saved_for_backward = tuple(
            [e for e in bound_symbols[-1].args[1][0] if hasattr(e, "name") and e.name in args_remained]
        )
        if isinstance(bound_symbols[-1].args[0]["output"], Sequence):
            output = tuple(
                [o for o in bound_symbols[-1].args[0]["output"] if hasattr(o, "name") and o.name in args_remained]
            )
        else:
            output = bound_symbols[-1].args[0]["output"]
        flat_output = tuple(
            [o for o in bound_symbols[-1].args[0]["flat_output"] if hasattr(o, "name") and o.name in args_remained]
        )
        new_dict = {"output": output, "flat_output": flat_output, "flat_args": bound_symbols[-1].args[0]["flat_args"]}

        # Create the new args and substitute return symbol
        bsym = bound_symbols[-1].from_bsym(args=(new_dict, (saved_for_backward, bound_symbols[-1].args[1][1])))
        bound_symbols[-1] = bsym
    # Bw trace
    else:

        def _returned(seq: Sequence) -> tuple:
            ret = []
            for e in seq:
                if e is None:
                    ret.append(None)
                elif isinstance(e, Sequence):
                    ret.append(_returned(e))
                elif isinstance(e, Proxy) and e.name in args_remained:
                    ret.append(e)
                elif not isinstance(e, Proxy):
                    raise RuntimeError(f"type not recognized: {type(e)}")

            return tuple(ret)

        # Backward output is a tuple, and generally a tuple of tuple (())
        original_returned = bound_symbols[-1].args
        returned = _returned(original_returned)
        bound_symbols[-1] = bound_symbols[-1].from_bsym(args=returned)

    extrace: TraceCtx = from_trace(trace)
    extrace.bound_symbols = bound_symbols
    return extrace


def map_executors_from_reduced_trace_to_complete_trace(
    complete_trace: TraceCtx, common_blocks: list[tuple], ex_mappings: list[Executor]
) -> list[Executor]:
    """
    Generate executors mappings (trace region -> executor) for the complete trace once the optimization has been performed on a reduced trace.

    This implementation currently relies on the fact that transformer blocks are contiguous in trace
    or they have a common gap region between them (in case for bw trace).

    The output executor list has size equal to the complete trace regions size.

    Args:
        complete_trace: A computation trace.
        common_blocks: A previously computed common block pattern.
        ex_mappings: The executor mappings for the reduce trace.
    """
    from thunder.executors.torchex import ex as torch_ex

    if len(common_blocks) < 2:
        raise AssertionError("No common block found")

    # Check that we maintain the pattern
    regions_between_blocks = _regions_between_blocks(complete_trace, common_blocks)

    # These are the trace region indices (referred to the complete trace) that we have excluded from the reduced trace optimization.
    # We have also to integrate their executors.
    # By default torchex will be used as currently no complex (optimizable) ops are present so far (they are usually reshape ops).
    indices_excluded: list = _indices_to_exclude_between_common_blocks(common_blocks)

    # Correctness assertion
    if regions_between_blocks:
        assert len(indices_excluded) % regions_between_blocks == 0
        assert len(indices_excluded) // regions_between_blocks == len(common_blocks) - 1

    # Solution starting point: copy up to the end of the first common block
    complete_trace_executors: list[Executor] = ex_mappings[: common_blocks[0][1] + 1]
    # Get the executors sequence to share from the first block to all the other equal blocks.
    to_share: list[Executor] = []
    for i in range(len(common_blocks) - 1):
        # First region bewteen block, adding here as this was not present in the reduce trace (not found in the ex_mappings structure)
        if i == 0:
            to_share.extend([torch_ex] * regions_between_blocks)

        to_share.extend(ex_mappings[common_blocks[0][0] : common_blocks[0][1] + 1])

        # We have to add back the excluded regions (see comment 15 lines above).
        if i < len(common_blocks) - 2:
            to_share.extend([torch_ex] * regions_between_blocks)

    # Extend by sharing mappings of transformer blocks
    complete_trace_executors.extend(to_share)
    # Extend with the remained bsyms
    complete_trace_executors.extend(ex_mappings[common_blocks[0][1] + 1 :])

    # Check that we have all the executors needed
    len_got = len(complete_trace_executors)
    len_expected = len(complete_trace.bound_symbols)
    if len_got != len_expected:
        raise AssertionError(
            f"Trace regions size is different from the obtained executors lenght: {len_expected} - {len_got}"
        )

    return complete_trace_executors


# This fn is used before compile data being set, rely on kwargs
def get_fw_bw_split_backends_options(bsym: BoundSymbol | None = None, **kwargs) -> list | dict:
    """
    Retrieves the executors tuning options for the vector jacobian product pass.
    These executors must be tuned at the vjp stage as we have to choose the correspective backward grad function.

    For new executors support the followig lists can be expanded.

    A guard is put for the transformer_engine_ex as its usage should not be tuned if not requested in a explicit way.

    Args:
        bsym: The query bound symbol.
    """
    from thunder.executors.sdpaex import sdpa_ex
    from thunder.executors.cudnnex import cudnn_ex
    from thunder.executors.fa3ex import fa3_ex
    from thunder.executors.transformer_engineex import transformer_engine_ex

    if kwargs is None or not kwargs.get("autotune_enable_te", False):
        options: dict[str, list] = {
            "scaled_dot_product_attention": [sdpa_ex, cudnn_ex, fa3_ex],
        }
    else:
        options: dict[str, list] = {
            "linear": [transformer_engine_ex],
            "scaled_dot_product_attention": [sdpa_ex, cudnn_ex, fa3_ex],
        }

    return options.get(bsym.sym.name, []) if bsym else options

def trace_symbolic_hash(trace: TraceCtx) -> str:
    res = ""
    for b in trace.bound_symbols:
        # Ignoring unpacks as when tuple has size zero, there are cases when None is given as static args/output and cases where a zero sized tuple is returned.
        res += symbol_hash(bsym=b, ignore_returns_meta=True, ignore_unpacks_meta=True)
    return res


supported_file_modes = set(['json'])
def dump_traces_placement(
    *,
    fw_trace: TraceCtx,
    bw_trace: TraceCtx,
    exs_fw: list[Executor],
    exs_bw: list[Executor],
    apply_remat: bool,
    file_name: str,
    output_mode: str = 'json'
) -> str:
    """
    Creates an output configuration file where the current forward and backward trace optimization are saved.

    Args:
        fw_trace: A forward trace.
        bw_trace: A backward trace.
        exs_fw: Forward trace region executors.
        exs_bw: Backward trace region executors.
        apply_remat: If forward and backward traces are output of rematerialize_forward_and_backward
        file_name: The output file name.
        output_mode: The output file format. Must be one of ['json'].
    """
    assert output_mode in supported_file_modes

    if output_mode == 'json':
        # We defined an unique trace by reading its bsym metadata, the proxies name are ignored as they may
        # change but the overall computation can remain the same.
        fw_hash = trace_symbolic_hash(fw_trace)
        bw_hash = trace_symbolic_hash(bw_trace)

        executors_fw_name = [ex.name if (ex and ex.name != 'empty') else "None" for ex in exs_fw]
        executors_bw_name = [ex.name if (ex and ex.name != 'empty') else "None" for ex in exs_bw]

        assert len(fw_trace.bound_symbols) == len(executors_fw_name)
        assert len(bw_trace.bound_symbols) == len(executors_bw_name)

        from thunder.backend_optimizer.optimizer import logger
        logger.info(
            f"Size match between len(fw_trace.bound_symbols)[{len(fw_trace.bound_symbols)}] and len(executors_fw_name)[{len(executors_fw_name)}]"
        )
        logger.info(
            f"Size match between len(bw_trace.bound_symbols)[{len(bw_trace.bound_symbols)}] and len(executors_bw_name)[{len(executors_bw_name)}]"
        )
        logger.info(f"Saving configuration in {file_name}")

        data = {
            "forward": {
                "hash": fw_hash,
                "executors": executors_fw_name,
            },
            "backward": {
                "hash": bw_hash,
                "executors": executors_bw_name,
            },
            "rematerialize": apply_remat
        }
        try:
            with open(file_name, "w") as file:
                import json
                json.dump(data, file)
        except Exception:
            from thunder.backend_optimizer.optimizer import logger
            import traceback
            err = traceback.format_exc()
            logger.error(f"Can not dump {file_name} file:\n{err}")
            return ""
        return file_name
    return ""

def apply_results_from_file(
    *, fw_trace: TraceCtx, bw_trace: TraceCtx, file: str, input_mode: str = "json"
) -> tuple[TraceCtx, TraceCtx]:
    """
    Generate a transformed forward and backward trace from a configuration file.
    Compatibility check is performed on both traces.

    Args:
        fw_trace: The original augmented forward trace.
        bw_trace: The original backward trace.
        file: The configuration file.
        input_mode: The configuration structure. Must be one of ['json'].
    """
    import json
    from thunder.executors.torchex import ex as torch_ex
    from thunder.executors.pythonex import ex as python_ex
    from thunder.executors.sdpaex import sdpa_ex
    from thunder.executors.cudnnex import cudnn_ex
    from thunder.executors.fa3ex import fa3_ex
    from thunder.executors.nvfuserex_impl import ex as nvfuser_ex
    from thunder.executors.torch_compile import torch_compile_ex
    from thunder.executors.torch_autograd import update_bw_from_forward_optimization

    assert input_mode in supported_file_modes

    # Extend this if more executors will be added
    conversion_map: dict[str | Hashable, Executor] = {
        'None': Executor('empty'),
        torch_ex.name: torch_ex,
        python_ex.name: python_ex,
        nvfuser_ex.name: nvfuser_ex,
        torch_compile_ex.name: torch_compile_ex,
        sdpa_ex.name: sdpa_ex,
        cudnn_ex.name: cudnn_ex,
        fa3_ex.name: fa3_ex
    }

    if input_mode == 'json':
        data = json.load(open(file, 'r'))

        fw_hash = trace_symbolic_hash(fw_trace)
        bw_hash = trace_symbolic_hash(bw_trace)
        assert fw_hash == data['forward']['hash']
        assert bw_hash == data['backward']['hash']

        fw_executors_recovered: list[str] = data["forward"]["executors"]
        extrace_fw = assign_executors(
            in_trace=fw_trace,
            executors_list=[conversion_map[ex] for ex in fw_executors_recovered],
            empty_str="empty",
            always_executors=get_always_executors(),
        )
        bw_executors_recovered: list[str] = data["backward"]["executors"]
        bw_trace = update_bw_from_forward_optimization(fw=extrace_fw, bw=bw_trace)
        extrace_bw = assign_executors(
            in_trace=bw_trace,
            executors_list=[conversion_map[ex] for ex in bw_executors_recovered],
            empty_str="empty",
            always_executors=get_always_executors(),
        )

        if data['rematerialize']:
            from thunder.core.rematerialization import rematerialize_forward_and_backward
            return rematerialize_forward_and_backward(extrace_fw, extrace_bw)
        return extrace_fw, extrace_bw
