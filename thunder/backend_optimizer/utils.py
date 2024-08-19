from collections.abc import Callable, Hashable, Sequence
from typing import Any
from thunder.core.dtypes import dtype, to_torch_dtype
from thunder.core.prims import PrimIDs
from thunder.core.proxies import CollectionProxy, FloatProxy, IntegerProxy, Proxy, TensorProxy, Variable, variableify
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx, get_tracectx, reset_tracectx, set_tracectx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor
from thunder.core.utils import check, safe_map_flat
import thunder.core.transforms as transforms
from itertools import chain
import torch
import thunder


def sequence_hash(s: Sequence) -> str:
    def rec(s) -> str:
        name = "["
        for e in s:
            if e is None:
                name += "None"
            elif hasattr(e, "name"):
                 name += e.name
            elif isinstance(e, Sequence):
                name += rec(e)
            elif isinstance(e, int):
                 name += 'int' + str(e)
            else:
                raise AssertionError(f"Unsupported type = {type(e)}")
        name += ']'
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


def get_not_used_intermediate_outsputs(trace_in: TraceCtx) -> list[TensorProxy]:
    def is_in_sequence(seq: Sequence[Any], t: TensorProxy):
        for e in seq:
            if isinstance(e, TensorProxy) and e.name == t.name:
                return True
        return False

    def is_possible_out(name: str):
        if not name.startswith("t"):
            return False
        num = name[1:]
        return num.isdigit()

    ans: list[TensorProxy] = []
    for b in trace_in.bound_symbols:
        f = False
        # Not a tensor
        if not isinstance(b.output, TensorProxy):
            continue
        # Not a produced tensor
        if not is_possible_out(b.output.name):
            continue
        for test in trace_in.bound_symbols:
            if (
                test.args is not None
                and (isinstance(test.args, tuple) or isinstance(test.args, list))
                and is_in_sequence(test.args, b.output)
            ):
                f = True
                break
        if not f:
            ans.append(b.output)
    return ans


def assign_executors(
    *,
    in_trace: TraceCtx,
    executor_list: list[Executor | FusionExecutor | OperatorExecutor]
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

        if len(executor_list) != len(in_trace.bound_symbols):
            raise AssertionError("len(executor_list) != len(in_trace.bound_symbols)")

        cached_subsymbols: dict[str, Sequence[BoundSymbol]] = {}
        executor_mapping: dict[str, Executor] = {}
        unique_fusion_executors = set()

        # Input should have equal length
        if len(executor_list) != len(in_trace.bound_symbols):
            raise AssertionError("len(executor_list) != len(extrace.bound_symbols)")

        for b, e in zip(in_trace.bound_symbols, executor_list):
            if isinstance(e, FusionExecutor):
                unique_fusion_executors.add(e)
            if isinstance(b.output, TensorProxy):
                executor_mapping[b.output.name] = e

        extrace = transforms.visitor_transform_paired(in_trace, visit, zip(in_trace.bound_symbols, executor_list))

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
            # Note some symbols may be cut out by the fusion pass -> CSE
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


def benchmark_trace(
    trace: TraceCtx,
    iters: int = 1,
    show_func=False,
    apply_del_last_used=True,
    snapshot=False,
    snapshot_name="",
    nvsight: bool = False,
    nvsight_fn_name: str = "",
) -> tuple[float, float, Any]:
    from thunder.executors.passes import del_last_used
    import inspect

    input_args = []

    if trace.bound_symbols[-1].sym.id != PrimIDs.RETURN:
        raise AssertionError("Missing return statement")

    def compute_time_cost_nvsight(fn: Callable, iters: int, *args) -> tuple[float, float, Any]:
        try:
            warm_up_iters = 50
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Warm up cycles
            for _ in range(warm_up_iters):
                fn(*args)
            # Benchmark
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(iters):
                torch.cuda.empty_cache()
                torch.cuda.nvtx.range_push(f"thunder benchmark fn:{nvsight_fn_name}, iter{i}")
                fn(*args)
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

            return float("inf"), float("inf"), None
        except Exception as e:
            import inspect

            trc = inspect.getsource(fn)
            print(f"#NVSIGHT FN EXECUTION FAILED:\n{trc}")
            raise e

    def compute_time_cost_ms(fn: Callable, repr: str, iters: int, *args) -> tuple[float, float, Any]:
        try:
            warm_up_iters = 50
            out = None

            # Warm up cycles
            for _ in range(warm_up_iters):
                fn(*args)
            # Snapshot request
            if snapshot:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.memory._record_memory_history()
                fn(*args)
                torch.cuda.memory._dump_snapshot(snapshot_name + "_benchmark.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
            # Benchmark
            stream = torch.cuda.current_stream()
            max_allocated_bytes = 0
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            torch.cuda.synchronize()
            for i in range(iters):
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                torch.cuda.empty_cache()
                torch.cuda._sleep(1_000_000)
                start_events[i].record(stream)
                fn(*args)
                end_events[i].record(stream)
                max_allocated_bytes = max(
                    max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device())
                )

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            # print(f"times: {times}")
            tot_time = sum(times) / iters
            return tot_time, max_allocated_bytes, out
        except Exception as e:
            print(f"#FN EXECUTION FAILED:\n{repr}")
            raise e

    def print_input_args(args, level=0, show_content=False):
        for e in args:
            if isinstance(e, tuple) or isinstance(e, list):
                print_input_args(e, level=level + 1)
            else:
                print(f"level {level}", type(e))

    # def print_trace_execution_output(out: Any, show_content=False):
    #     if isinstance(out, tuple):
    #         for e in out:
    #             print(f'{type(e)}')
    #     else:
    #         print(f'{type(out)}')

    # TODO (matteochen): convert this into dict
    def thunder_to_torch_float_dtype(tp: dtype, byte: int) -> torch.dtype:
        if byte == 1:
            raise AssertionError("Not implmented: 8 bit float")
        # Dispatch flaot 16 type 1 from type 2
        elif byte == 2:
            if tp._name == thunder.bfloat16._name:
                return torch.bfloat16
            else:
                return torch.float16
        elif byte == 4:
            return torch.float32
        elif byte == 8:
            return torch.float64
        else:
            raise AssertionError(f"Not supported byte = {byte}")

    # TODO (matteochen): convert this into dict
    def thunder_to_torch_int_dtype(byte: int) -> torch.dtype:
        if byte == 1:
            return torch.int8
        elif byte == 2:
            return torch.int16
        elif byte == 4:
            return torch.int32
        elif byte == 8:
            return torch.int64
        else:
            raise AssertionError(f"Not supported byte = {byte}")

    # TODO (matteochen): use more appropriate mock int and float
    def transform_input_tuple(t: tuple, level=0) -> tuple:
        res = []
        for e in t:
            if type(e) is tuple:
                res.append(transform_input_tuple(e, level + 1))
            else:
                if isinstance(e, TensorProxy):
                    res.append(transform_tensor(e))
                elif isinstance(e, IntegerProxy):
                    if e.python_type is bool:
                        res.append(False if e.value is None else e.value)
                    else:
                        res.append(0 if e.value is None else e.value)
                elif isinstance(e, FloatProxy):
                    res.append(0.0 if e.value is None else e.value)
                elif e is None:
                    res.append(None)
                else:
                    raise AssertionError(f"Input arg type not recognized: {type(e)}")
        return tuple(res)

    def transform_tensor(arg: TensorProxy) -> torch.Tensor:
        from thunder.core.dtypes import is_float_dtype, is_signedinteger_dtype, is_boolean_dtype

        # TODO (matteochen): Missing parallel and fsdp handling...
        # TODO (matteochen): Missing support for meta types ...
        dtype = arg.dtype
        shape = arg.shape
        device = arg.device
        requires_grad = arg.requires_grad

        torch_dtype = to_torch_dtype(dtype)
        if torch_dtype is None:
            raise AssertionError(f'Unrecognized thunder dtype: {dtype}')
        if is_float_dtype(dtype):
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

    # print(f'BENCHMARKING:\n{trace}')
    # def p(args):
    #     for e in args:
    #         if not isinstance(e, Sequence):
    #             if isinstance(e, torch.Tensor):
    #                 print(f'{e.size()}')
    #             else:
    #                 try:
    #                     print(f'{e.name} -> {e}')
    #                 except:
    #                     print(f'{e}')
    #         else:
    #             print('rec')
    #             p(e)
    # p(trace.args)
    # print('##################')
    # p(input_args)

    # Can we remove this check?
    # TODO (matteochen): use more appropriate mock int and float
    if isinstance(trace.args, Sequence):
        for arg in trace.args:
            if isinstance(arg, tuple):
                input_args.append(transform_input_tuple(arg))
            elif isinstance(arg, TensorProxy):
                e = transform_tensor(arg)
                input_args.append(e)
            elif isinstance(arg, IntegerProxy):
                if arg.python_type is bool:
                    input_args.append(False if arg.value is None else arg.value)
                else:
                    input_args.append(0 if arg.value is None else arg.value)
            elif isinstance(arg, FloatProxy):
                input_args.append(0.0 if arg.value is None else arg.value)
            else:
                raise AssertionError(f"Input arg type not recognized: {type(arg)}")
    else:
        raise AssertionError("Unexpexcted args type")

    if apply_del_last_used:
        trace = del_last_used(trace)

    trace_tok = set_tracectx(trace)

    # Obtain the python executable string
    executable_str = trace.python()
    executable = trace.python_callable()
    if show_func:
        print(inspect.getsource(executable))

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
