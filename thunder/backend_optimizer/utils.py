from collections.abc import Callable, Hashable, Sequence
from typing import Any
from thunder.core.compile_data import get_compile_data
from thunder.core.dtypes import to_torch_dtype
from thunder.core.prims import PrimIDs
from thunder.core.proxies import AnyProxy, FloatProxy, IntegerProxy, Proxy, TensorProxy, Variable, variableify
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx, get_tracectx, reset_tracectx, set_tracectx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor
from thunder.core.utils import check, safe_map_flat
import thunder.core.transforms as transforms
from itertools import chain
import torch


# Maybe we can use id(s)
def sequence_hash(s: Sequence) -> str:
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
    try:
        return ex.can_execute(bsym)
    except Exception:
        return False


def get_first_available_operator_executor(
    *, bsym: BoundSymbol, executors: Sequence[Executor], empty_hash: str = "empty"
):
    for ex in executors:
        if isinstance(ex, FusionExecutor):
            continue
        if can_executor_execute(ex, bsym):
            return ex
    return Executor(name=empty_hash)


def flatten_sequence(sequence: Sequence) -> list:
    res = []
    for e in sequence:
        if isinstance(e, Sequence):
            res.extend(flatten_sequence(e))
        # Skip Nones as they are not useful
        elif e is not None:
            res.append(e)
    return res


def get_not_used_intermediate_outsputs(trace_in: TraceCtx) -> list[Proxy]:
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
    # Currently this is O(max(len(bsym.output)) * N^2)
    # Can we check only bsym after the one in the outer loop in the inner loop (over trace.bound_symbols) ?
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
    from thunder.backend_optimizer.optimizer import log, LogLevel

    log(f"Returning not used proxies: {[p.name for p in ans]}", level=LogLevel.DEBUG)
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
                    # raise AssertionError('Failed to retrive key in mapping')
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


def operation_in_trace(*, trace: TraceCtx, op: str) -> bool:
    # Some optimizations are not available as symbols
    always_true = set(["bookend"])

    if op in always_true:
        return True
    for b in trace.bound_symbols:
        if b.sym.name == op:
            return True
    return False


def is_te_used(trace: TraceCtx) -> bool:
    from thunder.executors.transformer_engineex import linear_bound_symbol_name_prefix
    from thunder.executors.transformer_engineex import te_functional_linear_backward_name

    for bsym in trace.bound_symbols:
        if (
            bsym.sym.name.startswith(linear_bound_symbol_name_prefix)
            or bsym.sym.name == te_functional_linear_backward_name
        ):
            return True
    return False


def is_backward_trace(trace: TraceCtx) -> bool:
    sig = trace.signature_with_no_ctx()
    return sig.find("backward") >= 0


def benchmark_trace(
    trace: TraceCtx,
    iters: int = 1,
    show_func=False,
    apply_del_last_used=True,
    snapshot=False,
    snapshot_name="",
    nvsight: bool = False,
    nvsight_fn_name: str = "",
    **kwargs,
) -> tuple[float, float, Any]:
    from thunder.executors.passes import del_last_used
    import inspect

    torch.compiler.reset()

    def compute_time_cost_nvsight(fn: Callable, iters: int, *args) -> tuple[float, float, Any]:
        try:
            warm_up_iters = 50
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Warm up cycles
            for _ in range(warm_up_iters):
                # cloned_args = clone_args(args)
                fn(*args)
                # del cloned_args
            # Benchmark
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(iters):
                # cloned_args = clone_args(args)
                torch.cuda.empty_cache()
                torch.cuda.nvtx.range_push(f"thunder benchmark fn:{nvsight_fn_name}, iter{i}")
                fn(*args)
                torch.cuda.nvtx.range_pop()
                # del cloned_args
            torch.cuda.cudart().cudaProfilerStop()

            return float("inf"), float("inf"), None
        except Exception as e:
            import inspect

            trc = inspect.getsource(fn)
            print(f"#Trace execution failed for nvsight (error: {e}):\n{trc}")
            raise e

    def clone_args(args):
        res = []
        for arg in args:
            if isinstance(arg, Sequence):
                res.append(clone_args(arg))
            else:
                if isinstance(arg, torch.Tensor):
                    res.append(arg.clone().detach())
                else:
                    res.append(arg)
        return tuple(res)

    def compute_time_cost_ms(fn: Callable, repr: str, iters: int, *args) -> tuple[float, float, Any]:
        try:
            current_iter = 0
            warm_up_iters = 50
            out = None

            # print_args(args)

            # Warm up cycles
            for _ in range(warm_up_iters):
                # cloned_args = clone_args(args)
                out = fn(*args)
                # del cloned_args
            # Snapshot request
            if snapshot:
                # cloned_args = clone_args(args)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.memory._record_memory_history()
                fn(*args)
                torch.cuda.memory._dump_snapshot(snapshot_name + "_benchmark.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
                # del cloned_args
            # Benchmark
            stream = torch.cuda.current_stream()
            max_allocated_bytes = 0
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            torch.cuda.synchronize()
            for i in range(iters):
                current_iter = i
                # cloned_args = clone_args(args)
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                torch.cuda.empty_cache()
                torch.cuda._sleep(1_000_000)
                start_events[i].record(stream)
                fn(*args)
                end_events[i].record(stream)
                max_allocated_bytes = max(
                    max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device())
                )
                # del cloned_args

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            # print(f"times: {times}")
            tot_time = sum(times) / iters
            return tot_time, max_allocated_bytes, out
        except Exception as e:
            print(f"#Trace execution failed at iter {current_iter} (error: {e})\n{repr}")
            raise e

    def build_static_args(sequence: Sequence, **kwargs) -> list:
        return transform_proxy_to_torch(sequence, level=0, **kwargs)

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
            # They are indexed like [saved_for_bw, others...]
            # NOTE: This may change in the future
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

            See how the backward trace need t4 as argument recoveered from the static args
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
        if nvsight:
            t, m, answer = compute_time_cost_nvsight(executable, iters, *input_args)
        else:
            t, m, answer = compute_time_cost_ms(executable, executable_str, iters, *input_args)
    except Exception as e:
        import traceback

        ex_str = traceback.format_exc()
        print(ex_str)
        # https://github.com/Lightning-AI/lightning-thunder/issues/664
        # Seems that this patch never work ...
        if (
            "call_method UserDefinedObjectVariable(set) __contains__ [UserDefinedObjectVariable()] {}" in str(e)
            and not nvsight
        ):
            print(
                "Executing with torch compile no full graph (this might still fail), see: https://github.com/Lightning-AI/lightning-thunder/issues/664"
            )
            torch_compiled = torch.compile(executable, fullgraph=False)
            try:
                t, m, answer = compute_time_cost_ms(torch_compiled, executable_str, iters, *input_args)
            except Exception as e:
                print(f"Compiled trace execution still failed:\n{e}")
    finally:
        reset_tracectx(trace_tok)

    # Restore the autocast value to not mess up the input trace
    if te_used:
        trace._include_te_fp8_autocast = cached_te_fp8_autocast_value

    return t, m, answer


def register_impl_executor(ex: Executor, id: PrimIDs, fn: Callable, checker: Callable) -> None:
    if ex.name == "nvfuser":
        from thunder.executors.nvfuserex_impl import register_supported

        register_supported(id, fn, checker)


def recover_ex_from_compile_option(option: str) -> Executor:
    if option.startswith("nv"):
        from thunder.executors.nvfuserex_impl import ex

        return ex
    else:
        raise AssertionError(f"Compile option not recognized: {option}")


def wrap_fn_with_exeuctor_compile_option(option, fn: Callable | None = None, *args):
    from thunder.core import compile_data

    cd = compile_data.get_compile_data()
    if option is not None:
        # Update compile option context
        if cd is None:
            raise AssertionError("compile_data is None")
        # TODO: use getattr
        old_opt: bool | None = cd.compile_options.get(option.fusion_tag, None)
        new_opt = True if old_opt is None or old_opt is False else False
        cd.compile_options[option.fusion_tag] = new_opt
        # Register the impl for the executor in order to be able to execute the id
        register_impl_executor(
            recover_ex_from_compile_option(option.fusion_tag),
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
    print_nested_sequence(trace.args)


# Display nest sequence arguments
def print_nested_sequence(args, show_dicts=False):
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
                    f'{tabs}{name + ": " if name else ""}{type(arg)}{arg if isinstance(arg, dict) and show_dicts else ""} {tensor_shape if tensor_shape else ""} {dtype if dtype else ""}'
                )
        print(f"Level {level} end")

    _print(args, 0)
    print("###################################### Debug args\n")


def update_compile_options_executor_list_after_fw_bw_split() -> None:
    from thunder.backend_optimizer.optimizer import get_fw_bw_split_backends_options

    cd = get_compile_data()
    assert cd

    # Get all the possible options that the vjp_optimization pass will use
    options: dict = get_fw_bw_split_backends_options()
    executors_list = list(cd.executors_list)

    # Remove all the initial options
    for _, v in options.items():
        for ex in v:
            if ex in executors_list:
                executors_list.remove(ex)

    # Putting at the front even though order does not matter
    for ex in cd.compile_options["executors_placed_by_fw_bw_split"]:
        executors_list.insert(0, ex)

    # Assign new compilation executors options
    cd.executors_list = executors_list


def transform_tensor(arg: TensorProxy, **kwargs) -> torch.Tensor:
    from thunder.core.dtypes import is_float_dtype, is_signedinteger_dtype, is_boolean_dtype

    # TODO (matteochen): Missing parallel and fsdp handling...
    # TODO (matteochen): Missing support for meta types ...
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


def transform_proxy_to_torch(sequence: Sequence, level=0, **kwargs) -> tuple | list:
    from thunder.executors.transformer_engineex import Context as C

    res = []
    for e in sequence:
        if type(e) is tuple:
            res.append(transform_proxy_to_torch(e, level + 1, **kwargs))
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


def reorder_executors_list(executors: Sequence):
    from thunder.backend_optimizer.optimizer import get_fw_bw_split_backends_options
    from thunder.executors.torch_compile import torch_compile_ex
    from thunder.executors.nvfuserex_impl import ex as nvfuser_ex

    reordered = []
    options = get_fw_bw_split_backends_options()

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
