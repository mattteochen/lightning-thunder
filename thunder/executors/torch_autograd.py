from dataclasses import replace
from typing import TYPE_CHECKING

import torch

import thunder.core.utils as utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy, variableify
from thunder.core.pytree import tree_flatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx, from_trace, set_tracectx, reset_tracectx
from thunder.core.transform_common import replace_redundant_inputs
from thunder.extend import OperatorExecutor

if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface


def rename_bwd_trace_outputs(bwd_trace: TraceCtx, fwd_trace: TraceCtx) -> TraceCtx:
    """Have backward trace output tensor proxy names follow `grad_for_<param>` format.

    Since ``i``-th tensor proxy of backward trace's outputs is grad of ``i``-th tensor proxy of forward trace's inputs,
    this method looks up to forward trace's inputs to get the param name for each grad.

    Args:
        bwd_trace:
        fwd_trace:

    Returns:
        :class:`thunder.core.trace.TraceCtx`
    """

    # [note: why setting trace ctx?]
    # [`TensorProxy.replace_name`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L1221-L1223) calls
    # [`tensorproxy`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L1506-L1520)
    # which then calls `TensorProxy.__init__`. `TensorProxy.__init__` of course calls
    # [` Proxy.__init__`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L81-L86).
    # `Proxy`'s dunder init calls [`make_proxy_name`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L81-L86)
    # which depends on a tracectx.
    trace_tok = set_tracectx(bwd_trace)

    swap_map: dict[VariableInterface, TensorProxy] = {}
    bwd_outputs, _ = tree_flatten(bwd_trace.output)
    fwd_inputs, _ = tree_flatten((fwd_trace.args, fwd_trace.kwargs))

    utils.check(len(bwd_outputs) == len(fwd_inputs), lambda: f"{len(bwd_outputs)=}, {len(fwd_inputs)=}")

    for fwd_arg, bwd_out in zip(fwd_inputs, bwd_outputs):
        if isinstance(bwd_out, TensorProxy):
            swap_map[variableify(bwd_out)] = bwd_out.replace_name(f"grad_for_{fwd_arg.name}")
    reset_tracectx(trace_tok)

    renamed_bwd_trace = from_trace(bwd_trace)
    renamed_bwd_trace.bound_symbols = []

    bsym: BoundSymbol
    for bsym in bwd_trace.bound_symbols:
        renamed_bwd_trace.bound_symbols.append(bsym.from_bsym_swap_proxies(swap_map=swap_map))

    return renamed_bwd_trace


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, return_none_instead_of_grads, compiled_backward, saved_tensors, saved_other, flat_output, *flat_args
    ):
        # Here we just propagate the tensors through the autograd graph
        ctx.return_none_instead_of_grads = return_none_instead_of_grads
        ctx.saved_other = saved_other
        ctx.compiled_backward = compiled_backward

        # We must save tensors using ctx.save_for_backward
        ctx.save_for_backward(*saved_tensors)
        return flat_output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *args):
        # ctx.saved_tensors is a tuple of tensors saved in forward. Our compiled
        # backward is a really long function that takes all the tensors saved in
        # forward and gradually uses them to compute the gradients of the
        # inputs. Unfortunately, Python holds a reference to all arguments of a
        # function until the function returns, even if we delete the variable
        # "saved_tensors" inside the function, the tensors will still be held in
        # memory until the function returns. Fortunately, Python passes mutable
        # objects by reference, so we can just replace the saved_tensors with an
        # empty list and the memory will be freed immediately. We must also
        # delete the reference to the saved_tensors in the context, otherwise
        # the memory will be freed only when the context is deleted.
        saved_tensors_list = list(ctx.saved_tensors)  # Make a copy as we will mutate it

        # This is an undocumented API, but it's the only way to clear the
        # reference to the saved tensors in the context
        ctx.maybe_clear_saved_tensors()  # Delete the reference to all saved tensors in the context
        grads = ctx.compiled_backward([saved_tensors_list, ctx.saved_other], args)

        # Inside the compiled backward we must clear the saved_tensors_list
        assert not saved_tensors_list, "saved_tensors_list must be empty after calling compiled_backward"
        # TODO(crcrpar): Remove if-else once `dist_prims.stash_grad_for_fsdp` starts to return `None`
        # NOTE(crcrpar): In fsdp no-sync, unsharded gradients are attached and accumulated to their parameters as the attr of `_thunder_fsdp_unsharded_grad` in order to avoid shape mismatch of a param and its grad. When exiting the no_sync context, the accumulated, unsharded gradients are reduce-scattered into the attr of `grad` and `_thunder_fsdp_unsharded_grad` is removed.
        if not ctx.return_none_instead_of_grads:
            return (None, None, None, None, None, *grads)
        else:
            n_grads = len(grads)
            del grads
            return (None, None, None, None, None, *([None] * n_grads))

def split_forward_backward(computation_trc: TraceCtx, compile_data, compile_stats, autotune_type, /, *flat_args):
    from thunder.core.rematerialization import rematerialize_all_gather, rematerialize_forward_and_backward
    from thunder.core.transforms import forward_and_backward_from_trace
    from thunder.distributed.transforms import FSDPCommBucketing
    from thunder.distributed.utils import sort_data_parallel_syncs, sort_waits, sort_communication_ops
    from thunder.executors.passes import del_last_used, transform_for_execution, autotune_transform_for_execution
    from thunder.visualizer.visualizer_helper import Visualizer
    from thunder.backend_optimizer.optimizer import log, LogLevel, TraceType, BackendOptimizer, OptimizerType, benchmark_trace

    def  split():
        utils.check(compile_data is not None, lambda: "`compile_data` is required")
        # NOTE: This function is rather slow, so it's intended to be used
        # behind a cache.
        tensor_cls = (torch.Tensor, TensorProxy)
        requires_grad_mask = tuple(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
        # If none of the inputs require gradients, raise an error
        if not any(requires_grad_mask):
            raise RuntimeError("PyTorch's Autograd interface requires at least one tensor input with requires_grad=True")

        primal_trace = computation_trc
        primal_trace = sort_data_parallel_syncs(primal_trace)

        # Handled by the caller if autotune is not None
        if compile_stats is not None and autotune_type is None:
            compile_stats.last_traces.append(primal_trace)

        # torch.autograd.Function doesn't support non-flat outputs, the
        # grads wouldn't be propagated and backward receives None for each
        # non-flat non-tensor output. The output must also be a flat tuple,
        # not any other container type. So we need to flatten the outputs of
        # the forward trace and inputs of the backward trace.
        fw_trace, bw_trace = forward_and_backward_from_trace(primal_trace, torch_autograd=True)

        fw_traces = [fw_trace]
        bw_traces = [bw_trace]

        from thunder.distributed import FSDPType

        # only enable rematerialize_params_in_backward when using FSDP ZeRO3
        _rematerialize_params_in_backward = (
            getattr(compile_data.fn, "use_fsdp", False) and getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3
        )
        if _rematerialize_params_in_backward:
            fw_trace, bw_trace = rematerialize_all_gather(fw_trace, bw_trace)

        # Update the backward trace to only compute gradients for the
        # inputs that require gradients
        assert bw_trace.bound_symbols[-1].sym.id == PrimIDs.RETURN
        filtered_grads = tuple(
            (arg_grad if requires_grad else None)
            for arg_grad, requires_grad in utils.safe_zip(bw_trace.bound_symbols[-1].args[0], requires_grad_mask)
        )

        # autograd.Function.backward expects a flat tuple of gradients
        bw_trace.bound_symbols[-1] = replace(bw_trace.bound_symbols[-1], args=(filtered_grads,))

        _fsdp_comm_bucketing: FSDPCommBucketing | None = None
        if getattr(compile_data.fn, "use_fsdp", False):
            _fsdp_comm_bucketing = FSDPCommBucketing(compile_data, computation_trc)
            fw_trace = _fsdp_comm_bucketing.apply_bucketing_to_forward_trace(fw_trace)

        do_apply_bucketing_bw_trace: bool = getattr(compile_data.fn, "use_fsdp", False)

        # Now we can run the optimization passes on the forward trace
        visualizer = Visualizer(produce_hidden=False)
        backend_optimizer_ctx: BackendOptimizer | None = (
            None
            if autotune_type is None
            else BackendOptimizer(
                priority_executors=compile_data.executors_list,
                apply_bucketing_bw_trace=do_apply_bucketing_bw_trace,
                produce_log=True,
                visualizer=visualizer,
                optimizer_type=autotune_type,
            )
        )

        visualizer.set_fw_initial_trace(fw_trace)
        # Get optimzied fw trace
        fw_extrace = (
            transform_for_execution(fw_trace, executors_list=compile_data.executors_list)
            if autotune_type is None
            else autotune_transform_for_execution(
                optimizer_context=backend_optimizer_ctx, trace=fw_trace, trace_type=TraceType.FW
            )
        )

        # If in default mode, otherwise the best fw will be returned only at the end
        if autotune_type is None:
            fw_traces.append(fw_extrace)
            visualizer.set_fw_optimized_trace(fw_extrace)

            # NOTE: autotuner will take care of this
            # Some of the optimization passes change proxies in the trace and
            # any change in the forward trace must be reflected in the backward
            # trace.
            original_bw_saved_tensors_for_backward = bw_trace.args[0][0]
            new_fw_saved_tensors_for_backward = fw_extrace.output[1][0]
            swap_map = {
                variableify(x): y
                for x, y in zip(original_bw_saved_tensors_for_backward, new_fw_saved_tensors_for_backward)
                if variableify(x) != variableify(y)
            }
            new_bsyms = replace_redundant_inputs(swap_map, bw_trace.bound_symbols)
            # replace_redundant_inputs doesn't replace the output of
            # UNPACK_SEQUENCE so we do it manually. Here we have certain
            # assumptions about the structure of the backward trace.
            assert bw_trace.bound_symbols[0].sym.id == PrimIDs.UNPACK_TRIVIAL
            assert bw_trace.bound_symbols[0].kwargs["name"] == "saved_for_backward"
            assert bw_trace.bound_symbols[4].sym.id == PrimIDs.UNPACK_SEQUENCE
            assert bw_trace.bound_symbols[4].args[0].name == "C0"
            new_bsyms[4] = new_bsyms[4].from_bsym_swap_proxies(
                swap_map,
                skip_inputs=False,
                skip_output=False,
                skip_subsymbols=False,
            )
            bw_trace.bound_symbols = new_bsyms

            if do_apply_bucketing_bw_trace:
                bw_trace = _fsdp_comm_bucketing.apply_bucketing_to_backward_trace(bw_trace)

        # Now we can run the optimization passes on the backward trace
        # TODO Restore request for no rematerialization

        visualizer.set_bw_initial_trace(bw_trace)
        if autotune_type is not None:
            fw_extrace, bw_extrace = autotune_transform_for_execution(
                optimizer_context=backend_optimizer_ctx, trace=bw_trace, trace_type=TraceType.BW
            )
            fw_traces.append(fw_extrace)
            visualizer.set_bw_optimized_trace(fw_extrace)
        else:
            bw_extrace = transform_for_execution(
                bw_trace,
                executors_list=compile_data.executors_list,
            )
        bw_traces.append(bw_extrace)
        visualizer.set_bw_optimized_trace(bw_extrace)

        # TODO Restore request for no rematerialization
        c, m, _ = benchmark_trace(fw_extrace, iters=50)
        log(f'before remat fw trace time = {c}, mem = {m}', level=LogLevel.INFO)
        c, m, _ = benchmark_trace(bw_extrace, iters=50)
        log(f'before remat bw trace time = {c}, mem = {m}', level=LogLevel.INFO)
        fw_extrace, bw_extrace = rematerialize_forward_and_backward(fw_extrace, bw_extrace)
        c, m, _ = benchmark_trace(fw_extrace, iters=50)
        log(f'after remat fw trace time = {c}, mem = {m}', level=LogLevel.INFO)
        c, m, _ = benchmark_trace(bw_extrace, iters=50)
        log(f'after remat bw trace time = {c}, mem = {m}', level=LogLevel.INFO)
        fw_traces.append(fw_extrace)
        bw_traces.append(bw_extrace)

        # We need to sort the waits in forward and backward trace to overlap
        # computation with communication
        # For performance we need the wait_prim_impl nodes in the execution trace to be as far from the
        # communication ops as possible. But it causes the all_gather_prim_impl nodes gathered at the start of
        # backward trace and increases the peak allocated memory
        use_fsdp: bool = getattr(compile_data.fn, "use_fsdp", False)
        if use_fsdp:
            assert hasattr(compile_data.fn, "sharding_strategy")
            if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3:
                from thunder.distributed import FSDPBucketingStrategy
                from thunder.distributed.utils import limit_in_flight_allgathers

                fw_extrace = sort_communication_ops(fw_extrace)
                fw_extrace = limit_in_flight_allgathers(
                    fw_extrace,
                    3,
                    compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
                )
                bw_extrace = sort_communication_ops(bw_extrace)
                bw_extrace = limit_in_flight_allgathers(
                    bw_extrace,
                    3,
                    compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
                )
            if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO2:
                from thunder.distributed import FSDPBucketingStrategy
                from thunder.distributed.utils import limit_in_flight_allgathers
                from sys import maxsize as INT_MAX

                # sort the allgather+wait as consumer order just before consumer
                fw_extrace = sort_communication_ops(fw_extrace)
                # unlimited number of allgathers, i.e. allgathers are listed at the beginning of the trace in consumer order and wait stays just before wait
                fw_extrace = limit_in_flight_allgathers(
                    fw_extrace,
                    INT_MAX,
                    compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
                )
                bw_extrace = sort_waits(bw_extrace)
        use_ddp: bool = getattr(compile_data.fn, "use_ddp", False)
        if use_ddp:
            bw_extrace = sort_waits(bw_extrace)
        if (not use_ddp) and (not use_fsdp):
            from thunder.distributed.utils import maybe_sort_waits

            _, fw_extrace = maybe_sort_waits(fw_extrace)
            _, bw_extrace = maybe_sort_waits(bw_extrace)

        # Importing here to avoid cyclical dependencies in future.
        from thunder.executors.transformer_engineex import _transformer_engine_bwd_fp8_meta_sync, transformer_engine_ex

        if transformer_engine_ex in compile_data.executors_list:
            # NOTE: `_transformer_engine_bwd_fp8_meta_sync` may mutate `fw_extrace` or `bw_extrace`.
            _transformer_engine_bwd_fp8_meta_sync(fw_extrace, bw_extrace)

        fw_extrace = del_last_used(fw_extrace)
        fw_traces.append(fw_extrace)

        bw_extrace = del_last_used(bw_extrace, clear_mutable_collections=True)
        bw_traces.append(bw_extrace)

        bw_trace = rename_bwd_trace_outputs(bw_extrace, fw_extrace)

        # This is moved to the caller if autotune is enabled
        if compile_stats is not None and autotune_type is None:
            compile_stats.last_traces += fw_traces
            compile_stats.last_backward_traces += bw_traces

        # Enable wrapping with `te.fp8_autocast`.
        fw_extrace._include_te_fp8_autocast = True
        # We only want the forward function to be called with `te.fp8_autocast` manager.
        bw_extrace._include_te_fp8_autocast = False

        # Let's include the last traces also after all the passes
        visualizer.set_fw_final_trace(fw_extrace)
        visualizer.set_bw_final_trace(bw_extrace)

        # visualizer.produce()

        if autotune_type is None:
            return fw_extrace, bw_extrace
        else:
            return primal_trace, fw_extrace, bw_extrace, fw_traces, bw_traces

    # Defined executors that are matched inside the fw and bw split, hence outside the autotuner scope
    # TODO (matteochen): integrate Transofrmer Engine
    from thunder.executors.sdpaex import sdpa_ex
    from thunder.executors.cudnnex import cudnn_ex
    from thunder.executors.transformer_engineex import transformer_engine_ex

    executors_candidates: dict[str, list] = {
        'scaled_dot_product_attention': [sdpa_ex.name, cudnn_ex.name],
        'linear_layer': [transformer_engine_ex.name]
    }

    # TODO (matteochen): use BackendOptimizer tracing

    # If autotuner is enabled, we compare different impl of executors which are assigned inside the call 'forward_and_backward_from_trace'
    # as the autotuner will receive already split fw and bw traces
    if autotune_type is not None:
        cached_executor_list = list(compile_data.executors_list)
        is_tuned = False

        # We are interested to save the best_*s at the last iteration over the executors_candidates dict as the last
        # out *_extrace from calling split will contain all the best executors computed incrementally
        # i.e: best_* will track the best placemet for iteration (executors_candidates iteration) i plus every iteration from [0, i-1]
        best_cost: float = float('inf')
        best_fw_extrace: TraceCtx | None = None
        best_bw_extrace: TraceCtx | None = None
        best_fw_traces: list[TraceCtx] = []
        best_bw_traces: list[TraceCtx] = []
        best_primal_trace: TraceCtx | None = None
        best_executor: OperatorExecutor | None = None

        for i, (ex_type, ex_list) in enumerate(executors_candidates.items()):
            log(
                    f"================================================================================ Before Autotune Tuning: Optimizing {ex_type}",
                    level=LogLevel.DEBUG)
            # Search in the requested executor list if one or more than one options for a know multiple executable region is available
            to_benchmark = [ex for ex in cached_executor_list if ex.name in ex_list]

            if not to_benchmark:
                log(
                        f"================================================================================ Before Autotune Tuning: Skipping optimization for {ex_type} as not requested.",
                        level=LogLevel.DEBUG)

            for e in to_benchmark:
                compile_data.executors_list = [ex for ex in cached_executor_list if ex not in to_benchmark]
                compile_data.executors_list.insert(0, e)
                log(
                        f"================================================================================ Before Autotune Tuning: Testing compile data executors: {compile_data.executors_list}", level=LogLevel.DEBUG)

                primal_trace, fw_extrace, bw_extrace, fw_traces, bw_traces = split()
                time_fw, mem_fw, _ = benchmark_trace(fw_extrace, iters=10, apply_del_last_used=False)
                time_bw, mem_bw, _ = benchmark_trace(bw_extrace, iters=10, apply_del_last_used=False)
                tot_time = time_fw + time_bw
                tot_mem = mem_fw + mem_bw
                log(
                        f"================================================================================ Before Autotune Tuning: Benchmark {ex_type} options: {e.name}. Time fw = {time_fw} ms - Time bw = {time_bw} ms - Mem fw = {mem_fw / (2**30)} GB - Mem bw = {mem_bw / (2**30)} GB", level=LogLevel.DEBUG)
                log(
                        f"================================================================================ Before Autotune Tuning: Benchmark {ex_type} options: {e.name}. Time = {tot_time} ms - Mem = {tot_mem / (2**30)} GB", level=LogLevel.DEBUG)
                log(f'Fw trace:\n{fw_extrace}', level=LogLevel.DEBUG)
                log(f'Bw trace:\n{bw_extrace}', level=LogLevel.DEBUG)

                benchmark_cost = tot_time if autotune_type == OptimizerType.RUNTIME else tot_mem
                if benchmark_cost < best_cost:
                    is_tuned = True
                    best_cost = benchmark_cost
                    best_fw_extrace = fw_extrace
                    best_bw_extrace = bw_extrace
                    best_fw_traces = fw_traces
                    best_bw_traces = bw_traces
                    best_primal_trace = primal_trace
                    best_executor = e

                    # c, m , _ = benchmark_trace(best_fw_extrace, iters=10, apply_del_last_used=False)
                    # print(f'inside update {c}')
                    # c, m , _ = benchmark_trace(best_bw_extrace, iters=10, apply_del_last_used=False)
                    # print(f'inside update {c}')

            c, m , _ = benchmark_trace(best_fw_extrace, iters=10, apply_del_last_used=False)
            log(
                    f"================================================================================ Before Autotune Tuning: Benchmark {ex_type} best fw_extrace (time = {c}, mem = {m}):\n{best_fw_extrace}", level=LogLevel.DEBUG)
            c, m , _ = benchmark_trace(best_bw_extrace, iters=10, apply_del_last_used=False)
            log(
                    f"================================================================================ Before Autotune Tuning: Benchmark {ex_type} best bw_extrace (time = {c}, mem = {m}):\n{best_bw_extrace}", level=LogLevel.DEBUG)

            # Update the executor list with the winner executor for the current ex_type
            cached_executor_list = [ex for ex in cached_executor_list if ex not in to_benchmark]
            # We have a solution, we don't have it if not requested from the executor list
            if best_executor is not None:
                cached_executor_list.insert(0, best_executor)
            best_executor = None
            log(
                    f"================================================================================ Before Autotune Tuning: Benchmark {ex_type}, new executor list: {cached_executor_list}", level=LogLevel.DEBUG)

            # Update the compile stats on the last iter
            if i == len(executors_candidates)-1:
                # Check that we have solution, we don't have it if not requested from the executor list
                if is_tuned:
                    # Restore
                    compile_data.executors_list = list(cached_executor_list)

                    log(
                            f"================================================================================ Before Autotune Tuning: autotuned split_forward_backward from {executors_candidates}", level=LogLevel.DEBUG)
                    if compile_stats is not None:
                        compile_stats.last_traces.append(best_primal_trace)
                        compile_stats.last_traces += best_fw_traces
                        compile_stats.last_backward_traces += best_bw_traces

                    return best_fw_extrace, best_bw_extrace
                # If no solution is found at this optmization step, we proceed normally
                else:
                    # Restore before calling split
                    compile_data.executors_list = list(cached_executor_list)

                    log(
                            f"================================================================================ Before Autotune Tuning: not autotuned split_forward_backward from {executors_candidates}", level=LogLevel.DEBUG)
                    primal_trace, fw_extrace, bw_extrace, fw_traces, bw_traces = split()
                    if compile_stats is not None:
                        compile_stats.last_traces.append(primal_trace)
                        compile_stats.last_traces += fw_traces
                        compile_stats.last_backward_traces += bw_traces

                    return fw_extrace, bw_extrace
    else:
        return split()
