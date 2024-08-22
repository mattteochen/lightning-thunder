from collections.abc import Callable, Sequence
from enum import Enum
from thunder.backend_optimizer.utils import operation_in_trace, wrap_fn_with_exeuctor_compile_option
from thunder.core.prims import PrimIDs
from thunder.core.proxies import CollectionProxy, FloatProxy, IntegerProxy, TensorProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceCtx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_always_executors
from thunder.visualizer.visualizer_helper import Visualizer
from typing import Hashable
from thunder.backend_optimizer.utils import benchmark_trace

# Defining a wrapper fn as the imports will crash in the global scope
def get_fw_bw_split_backends_options(bsym: BoundSymbol | None = None) -> list | dict:
    from thunder.executors.sdpaex import sdpa_ex
    from thunder.executors.cudnnex import cudnn_ex
    from thunder.executors.fa3ex import fa3_ex
    from thunder.executors.transformer_engineex import transformer_engine_ex
    #Current configuration
    options: dict[str, list] = {
        # TODO: filter out TE only if requested
        'linear': [transformer_engine_ex],
        'scaled_dot_product_attention': [sdpa_ex, cudnn_ex, fa3_ex],
    }

    return options.get(bsym.sym.name, []) if bsym else options


class BenchmarkResult:
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
    MEMORY = 0
    RUNTIME = 1


class TraceType(Enum):
    FW = 0
    BW = 1


class OptimizationAlgorithm(Enum):
    BEST_FUSER = 0


class FusionCompileOptionsHelper:
    def __init__(self, fusion_tag: str, symbol_tag: str, id: PrimIDs, impl: Callable, checker: Callable) -> None:
        self.fusion_tag = fusion_tag
        self.symbol_tag = symbol_tag
        self.id: PrimIDs = id
        self.impl: Callable = impl
        self.checker: Callable = checker


class TraceCandidate:
    def __init__(self, *, trace: TraceCtx, compile_opt: FusionCompileOptionsHelper | None = None, label: str) -> None:
        self.trace: TraceCtx = trace
        self.compile_opt: FusionCompileOptionsHelper | None = compile_opt
        self.label: str = label


class TraceCandidates:
    def __init__(
        self,
        best_time: TraceCtx | None = None,
        best_mem: TraceCtx | None = None,
        compile_opt_time: FusionCompileOptionsHelper | None = None,
        compile_opt_mem: FusionCompileOptionsHelper | None = None,
    ) -> None:
        self.best_time: TraceCtx | None = best_time
        self.best_mem: TraceCtx | None = best_mem
        self.compile_opt_time: FusionCompileOptionsHelper | None = compile_opt_time
        self.compile_opt_mem: FusionCompileOptionsHelper | None = compile_opt_mem

    def __repr__(self) -> str:
        return f"\nBest runtime candidate:\n{self.best_time}\nBest memory candidate:\n{self.best_mem}"

    def is_set(self) -> bool:
        return False if self.best_time is None or self.best_mem is None else True

    def attach_best_time_candidate(self, trace: TraceCtx):
        self.best_time = trace

    def attach_best_mem_candidate(self, trace: TraceCtx):
        self.best_mem = trace

    def iterable(self) -> tuple[TraceCtx | None, TraceCtx | None]:
        return self.best_time, self.best_mem

    def compile_opt_iterables(self) -> tuple[FusionCompileOptionsHelper | None, FusionCompileOptionsHelper | None]:
        return self.compile_opt_time, self.compile_opt_mem


class OutputCandidate:
    def __init__(
        self, *, fw: TraceCtx, bw: TraceCtx, compile_opt: FusionCompileOptionsHelper | None = None, cost: float = 0.0
    ) -> None:
        self.fw: TraceCtx = fw
        self.bw: TraceCtx = bw
        self.compile_opt: FusionCompileOptionsHelper | None = compile_opt
        self.tot_cost: float = cost

    def __repr__(self) -> str:
        return f"Final output candidate: forward trace:\n{self.fw.__repr__()}\nFinal output candidate: backward trace:\n{self.bw.__repr__()}"


# Benchmark only traces will contain traces after the rematerialization call with fw and bw calls, reproducing what will be the real traces after the autotune pass
# Non benchmark traces will contain traces after the placement (default) with no call to remat
# We have duplciated those in order to maintain thunder compilation flow as the output from the autotuner will be the traces with no pass through rematerialization
# TODO: torchcompile_cat currently is not supported as the autotuner search space in the FusionExecutor section is limited to 1
class FusionStratHelper:
    def __init__(self) -> None:
        self.supported_executors: set = set(["nvfuser", "torchcompile"])
        self.optimized_traces_mem: list[dict[str | Hashable, tuple[TraceCtx, FusionCompileOptionsHelper | None]]] = []
        self.optimized_traces_mem_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_time: list[dict[str | Hashable, tuple[TraceCtx, FusionCompileOptionsHelper | None]]] = []
        self.optimized_traces_time_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []


class FusionExecutorsPlacementCtx:
    def __init__(self, *, placement: list, compile_options: FusionCompileOptionsHelper | None = None) -> None:
        self.placement: list = placement
        self.compile_options: FusionCompileOptionsHelper | None = compile_options


class ExecutorPlacementOptions:
    def __init__(self) -> None:
        self.placement_options_mem: list[FusionExecutorsPlacementCtx] = []
        self.placement_options_time: list[FusionExecutorsPlacementCtx] = []


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


log_level: LogLevel = LogLevel.INFO


def log(what: str, level: LogLevel = LogLevel.INFO):
    if log_level == LogLevel.DEBUG or log_level == level:
        print(f"================================================================================ Autotune: {what}")

class PlacerBase:
    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log: bool = True,
        apply_bucketing_bw_trace: bool,
        log_file_name: str,
        visualizer: Visualizer | None = None,
        optimizer_type: OptimizerType = OptimizerType.RUNTIME,
        compile_data,
        ) -> None:
        self.always_executors: tuple[Executor, ...] = get_always_executors()
        self.empty_executor_hashable_placeholder: str = "empty"
        self.executors: Sequence[Executor] = priority_executors
        self.fusion_executors: Sequence[FusionExecutor] = [
            ex for ex in self.executors if isinstance(ex, FusionExecutor)
        ]
        # Helper needed for later
        self.fusion_executors_saved_for_later: Sequence[FusionExecutor] = []

        self.debug_msg: str = ""
        self.partial_costs: dict[TraceCtx, float] = {}
        self.visualizer: Visualizer | None = visualizer
        self.log_file_name: str = log_file_name
        self.produce_log: bool = produce_log

        self.optimizer_type: OptimizerType = optimizer_type

        self.active_fw_trace_ctx: tuple[TraceCtx | None, FusionCompileOptionsHelper | None] = None, None
        self.cached_fw_traces: list[TraceCandidate] = []
        self.bw_trace_candidates: TraceCandidates = TraceCandidates()
        self.out_traces_candidates: list[OutputCandidate] = []
        self.best_pair_runtime: OutputCandidate
        self.best_pair_memory: OutputCandidate

        self.apply_bucketing_bw_trace: bool = apply_bucketing_bw_trace

        self.benchmark_iters: int = 20

        self.compile_data = compile_data

    def optimize(self):
        pass

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType):
        pass

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        return []

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        return (TraceCtx(), TraceCtx())

class FusionPlacer_BeamSearch(PlacerBase):
    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log: bool = True,
        apply_bucketing_bw_trace: bool,
        log_file_name: str,
        visualizer: Visualizer | None = None,
        optimizer_type: OptimizerType = OptimizerType.RUNTIME,
        compile_data,
    ) -> None:
        super().__init__(
            priority_executors=priority_executors,
            produce_log=produce_log,
            apply_bucketing_bw_trace=apply_bucketing_bw_trace,
            log_file_name=log_file_name,
            visualizer=visualizer,
            optimizer_type=optimizer_type,
            compile_data=compile_data
        )

        # Strat fusion
        self.fusion_strat_helper: FusionStratHelper = FusionStratHelper()
        self.executor_placement_options: ExecutorPlacementOptions = ExecutorPlacementOptions()

        from thunder.executors.nvfuserex_impl import linear, _linear_check
        from thunder.executors.nvfuserex_impl import matmul, _matmul_check
        self.known_fusion_ex_compile_options: dict[str | Hashable, list[FusionCompileOptionsHelper]] = {
            "nvfuser": [
                FusionCompileOptionsHelper("nv_enable_linear", "linear", PrimIDs.LINEAR, linear, _linear_check),
                FusionCompileOptionsHelper("nv_enable_matmul", "matmul", PrimIDs.MATMUL, matmul, _matmul_check),
                # FusionCompileOptionsHelper("nv_enable_bookend", "bookend"),
            ]
        }

    """
    ################################################## Internal methods ##################################################
    """

    def _best_runtime_and_memory_candidates(self, candidates):
        from thunder.core.rematerialization import rematerialize_forward_and_backward
        from thunder.backend_optimizer.utils import benchmark_trace

        min_value_time: float = float("inf")
        min_value_mem: float = float("inf")
        best_pair_runtime: OutputCandidate
        best_pair_memory: OutputCandidate
        pair: OutputCandidate
        for pair in candidates:
            if pair.compile_opt:
                remat_fw, remat_bw = wrap_fn_with_exeuctor_compile_option(
                    pair.compile_opt, rematerialize_forward_and_backward, pair.fw, pair.bw
                )
            else:
                remat_fw, remat_bw = rematerialize_forward_and_backward(pair.fw, pair.bw)
            # Create pair final options by applying final optimizations: cudagraphs and rematerialization
            pair_options: list[tuple[TraceCtx, TraceCtx]] = [
                (pair.fw, pair.bw),
                (remat_fw, remat_bw),
            ]
            # We want to verify that it is not set to false
            if self.compile_data.use_cudagraphs is None or self.compile_data.use_cudagraphs == True:
                from thunder.executors.cudagraphex import cudagraphex
                pair_options.extend(
                    [
                        (cudagraphex.fusion_pass(pair.fw), cudagraphex.fusion_pass(pair.bw)),
                        (cudagraphex.fusion_pass(remat_fw), cudagraphex.fusion_pass(remat_bw)),
                    ]
                )
            # Select the best options
            for pair_option in pair_options:
                fw = pair_option[0]
                bw = pair_option[1]

                pair_cost_time = 0
                pair_cost_mem = 0
                t, m, _ = benchmark_trace(fw, iters=self.benchmark_iters)
                # log(f"Pair fw time: {t} ms, mem: {m/(2**30)} GB", level=LogLevel.INFO)
                pair_cost_time = pair_cost_time + t
                pair_cost_mem = pair_cost_mem + m
                t, m, _ = benchmark_trace(bw, iters=self.benchmark_iters, fw_trace=fw)
                # log(f"Pair bw time: {t} ms, mem: {m/(2**30)} GB", level=LogLevel.INFO)
                pair_cost_time = pair_cost_time + t
                pair_cost_mem = pair_cost_mem + m

                if pair_cost_time < min_value_time:
                    best_pair_runtime = OutputCandidate(fw=fw, bw=bw, cost=pair_cost_time)
                    # log(f"New best runtime pair (no remat):\n{best_pair_runtime}", level=LogLevel.INFO)
                    min_value_time = pair_cost_time

                if pair_cost_mem < min_value_mem:
                    best_pair_memory = OutputCandidate(fw=fw, bw=bw, cost=pair_cost_mem)
                    # log(f"New best memory pair (no remat):\n{best_pair_memory}", level=LogLevel.INFO)
                    min_value_mem = pair_cost_mem

        return best_pair_runtime, best_pair_memory

    def _filter_candidates(self):
        self.debug_msg += "Traces benchmarks:\n\n"

        # We cache every optimized fw traces as they might impact differently on the bw trace
        # Number of fw traces to cached are: #fusion_executors * 2
        def fw_benchmark():
            # The optimizator builds the results in order following the self.fusion_executors list order
            pair_time: dict
            pair_mem: dict
            for pair_time, pair_mem in zip(
                self.fusion_strat_helper.optimized_traces_time, self.fusion_strat_helper.optimized_traces_mem
            ):
                trc_time, compile_opt_time = list(pair_time.values())[0]
                trc_mem, compile_opt_mem = list(pair_mem.values())[0]
                label = list(pair_time.keys())[0]
                # TODO (matteochen): remove the benchmark here as will done later on the bw pass
                c, m, _ = benchmark_trace(trc_time, self.benchmark_iters)
                # log(
                #     f'Benchmark fw end: Trace = [{label}] (time = {c} ms, mem = {m / (2**30)} GB)":\n{trc_time}',
                #     level=LogLevel.INFO,
                # )
                self.debug_msg += (
                    f"Trace name = [{label}] - Target: TIME - Time = {c} ms - Mem = {m / (2**30)} GB\n{trc_time}\n\n"
                )
                c, m, _ = benchmark_trace(trc_mem, self.benchmark_iters)
                # log(
                #     f'Benchmark fw end: Trace = [{label}] (time = {c} ms, mem = {m / (2**30)} GB)":\n{trc_mem}',
                #     level=LogLevel.INFO,
                # )
                self.debug_msg += (
                    f"Trace name = [{label}] - Target: MEM - Mem = {m / (2**30)} GB - Time = {c} ms\n{trc_mem}\n\n"
                )
                # For forward trace we cache the best placement for both runtime and memory for the current Fusion executor (represented by label)
                # if compile_opt_time is not None:
                #     print(f"Caching fw with compile options time: {compile_opt_time.fusion_tag}")
                # if compile_opt_mem is not None:
                #     print(f"Caching fw with compile options mem: {compile_opt_mem.fusion_tag}")

                for t, o in zip([trc_time, trc_mem], [compile_opt_time, compile_opt_mem]):
                    log(f'Caching fw candidate [compile option: {o.fusion_tag if o else "None"}]')
                    self.cached_fw_traces.append(
                        TraceCandidate(trace=t, compile_opt=o, label=label + '_enabled_' + o.fusion_tag if o is not None else label)
                    )

        def bw_benchmark():
            time_result = BenchmarkResult()
            memory_result = BenchmarkResult()

            # Find best trace for runtime
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_time_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]
                trace_time, trace_mem, res = benchmark_trace(trace, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0])
                self.debug_msg += f"Trace name = [{label}] - Target: TIME - Time = {trace_time} ms - Mem = {trace_mem / (2**30)} GB\n{trace}\n\n"
                # log(
                #     f'Benchmark trace (target TIME) "{label}" (time = {trace_time} ms, mem = {trace_mem / (2**30)} GB:\n{trace}',
                #     level=LogLevel.INFO,
                # )
                if trace_time < time_result.runtime:
                    time_result = BenchmarkResult(time=trace_time, memory=trace_mem, trace=trace, label=label, index=i)

            # Find best trace for memory
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_mem_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]

                trace_time, trace_mem, res = benchmark_trace(trace, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0])
                del res
                self.debug_msg += f"Trace name = [{label}] - Target: MEM - Mem = {trace_mem / (2**30)} GB - Time = {trace_time} ms\n{trace}\n\n"
                # log(
                #     f'Benchmark trace (target MEM) "{label}" (time = {trace_time} ms, mem = {trace_mem / (2**30)} GB:\n{trace}',
                #     level=LogLevel.INFO,
                # )
                if trace_mem < memory_result.memory:
                    memory_result = BenchmarkResult(
                        time=trace_time, memory=trace_mem, trace=trace, label=label, index=i
                    )

            # log(
            #     f'Benchmark end: Best trace time "{time_result.label} (time = {time_result.runtime} ms)":\n{time_result.trace}',
            #     level=LogLevel.INFO,
            # )
            # log(
            #     f'Benchmark end: Best trace mem "{memory_result.label} (mem = {memory_result.memory / (2 ** 30)} GB)":\n{memory_result.trace}',
            #     level=LogLevel.INFO,
            # )

            # Here we have to recover the traces without the pass through remat in order to be compliant
            # with thunder flow as we might have request for no remat
            # Unpack dict
            trc = list(self.fusion_strat_helper.optimized_traces_time[time_result.index].values())[0][0]
            self.bw_trace_candidates.attach_best_time_candidate(trc)
            trc = list(self.fusion_strat_helper.optimized_traces_mem[memory_result.index].values())[0][0]
            self.bw_trace_candidates.attach_best_mem_candidate(trc)

            # Now, finally build the pair fw and bw traces
            # The current fw trace is set by the caller and we take it as is. All current bw traces optimizations are made with the fw trace set by the caller
            for bw in self.bw_trace_candidates.iterable():
                self.out_traces_candidates.append(
                    OutputCandidate(fw=self.active_fw_trace_ctx[0], bw=bw, compile_opt=self.active_fw_trace_ctx[1])
                )

        match self.trace_type:
            case TraceType.FW:
                fw_benchmark()
            case TraceType.BW:
                bw_benchmark()

        if self.produce_log:
            import time

            timestamp: str = str(time.time())
            with open(f"{timestamp}-{self.log_file_name}", "w") as file:
                file.write(self.debug_msg)
                file.close()

            self.debug_msg = ""

    def _search_candidates(self, increment_factor: int = 1):
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols
        from thunder.core.rematerialization import rematerialize_forward_and_backward
        from thunder.backend_optimizer.utils import (
            get_not_used_intermediate_outsputs,
            sequence_hash,
            can_executor_execute,
            get_first_available_operator_executor,
            assign_executors,
        )

        def get_placed_trace(mapping: dict[str, Executor], bound_symbols_in: Sequence[BoundSymbol]):
            trc = from_trace(self.trace)
            trc.bound_symbols = list(bound_symbols_in)

            # For this partial trace we have to return all not used tensors otherwise the dce will cut them out
            tensors = get_not_used_intermediate_outsputs(trc)

            forced_return_bsym = self.trace.bound_symbols[-1].from_bsym(args=tensors)

            executor_configuration = []
            empty_executor = Executor(name=self.empty_executor_hashable_placeholder)
            keys = []
            for bsym in trc.bound_symbols:
                if bsym.sym.name == "return":
                    raise AssertionError("Return statement should not be here")
                    # executor_configuration.append(empty_executor)
                    # keys.append('return')
                elif isinstance(bsym.output, Sequence):
                    seq_hash = sequence_hash(bsym.output)
                    executor_configuration.append(mapping.get(seq_hash, empty_executor))
                    keys.append(seq_hash)
                elif (
                    isinstance(bsym.output, CollectionProxy)
                    or isinstance(bsym.output, TensorProxy)
                    or isinstance(bsym.output, IntegerProxy)
                    or isinstance(bsym.output, FloatProxy)
                ):
                    if bsym.output.name not in mapping:
                        raise AssertionError(f"Expected key {bsym.output.name} in mapping {mapping}")
                    executor_configuration.append(mapping[bsym.output.name])
                    keys.append(bsym.output.name)
                else:
                    raise AssertionError(f"Type not handled: {type(bsym.output)}")

            if trc.bound_symbols[-1].sym.name != "return":
                trc.bound_symbols.append(forced_return_bsym)
                executor_configuration.append(Executor(name=self.empty_executor_hashable_placeholder))
                keys.append("return")

            if len(trc.bound_symbols) != len(executor_configuration) or len(keys) != len(executor_configuration):
                raise AssertionError(
                    f"len trc.bound_symbols ({len(trc.bound_symbols)}) != len executor_configuration ({len(executor_configuration)}) != len keys ({len(keys)})"
                )

            placed_trace = assign_executors(
                in_trace=trc,
                executor_list=executor_configuration,
                always_executors=self.always_executors,
                empty_str=self.empty_executor_hashable_placeholder,
            )
            return placed_trace, keys, executor_configuration

        def _search(ex: FusionExecutor, executor_compile_option: FusionCompileOptionsHelper | None = None):
            """
            Fusable fn definition for nvFuser
            """
            # Each executor has a custom should fuse function, but the current impl need to access local executor object
            def _should_fuse_nvfuser(a: Node, b: Node):
                def _can_fuse_node(n: Node):
                    # if already merged, then node can be fused
                    if len(n.group_bsyms) > 1:
                        return True
                    bsym: BoundSymbol = n.group_bsyms[0]
                    can_fuse: bool = ex.can_fuse(bsym)
                    cuda_in_or_out: bool = ex.has_cuda_input_or_output(bsym)
                    return can_fuse and cuda_in_or_out

                return _can_fuse_node(a) and _can_fuse_node(b)

            """
            Fusable fn definition for torch.compile
            """
            def _should_fuse_torchcompile(a: Node, b: Node):
                def _can_fuse_node(n: Node):
                    if len(n.group_bsyms) > 1:
                        return True
                    bsym: BoundSymbol = n.group_bsyms[0]
                    return ex.can_fuse(bsym)

                return _can_fuse_node(a) and _can_fuse_node(b)

            def match_bsym_output(bsym_in: BoundSymbol, dicts: list[dict], ex_in: Executor):
                if isinstance(bsym_in.output, Sequence):
                    for d in dicts:
                        d[sequence_hash(bsym_in.output)] = ex_in
                elif (
                    isinstance(bsym_in.output, CollectionProxy)
                    or isinstance(bsym_in.output, TensorProxy)
                    or isinstance(bsym_in.output, IntegerProxy)
                    or isinstance(bsym_in.output, FloatProxy)
                ):
                    for d in dicts:
                        d[bsym_in.output.name] = ex_in
                else:
                    raise AssertionError(f"Type not handled: {type(bsym_in.output)}")

            merge_fn: Callable
            match ex.name:
                case 'nvfuser':
                    merge_fn = _should_fuse_nvfuser
                case 'torchcompile':
                    merge_fn = _should_fuse_torchcompile
            bound_symbol_groups = fuse_bound_symbols(
                self.trace, merge_fn
            )
            log(f"Num of groups = {len(bound_symbol_groups)}", level=LogLevel.DEBUG)

            for id, group in enumerate(bound_symbol_groups):
                log(f"Group id: {id}", level=LogLevel.DEBUG)
                for sub in group:
                    log(f"{sub.sym.name} -> out: {sub.output}", level=LogLevel.DEBUG)
                if log_level == LogLevel.DEBUG and len(group) > 0:
                    print("\n")

            dict_time_strat: dict[str, Executor] = {}
            dict_mem_strat: dict[str, Executor] = {}
            increasing_symbols = []
            for group_id, group in enumerate(bound_symbol_groups):
                log(f"Group id: {group_id}", level=LogLevel.DEBUG)
                log(f"group start = {group[0].sym.name}", level=LogLevel.DEBUG)
                log(f"group end = {group[-1].sym.name}", level=LogLevel.DEBUG)

                if group[0].sym.name != "return":
                    increasing_symbols += group

                # Is not a fusion region, get the optimal executor (OperatorExecutor)
                if len(group) < 2:
                    current_bsym = group[0]
                    log(f"--> Single group: {current_bsym.sym.name}", level=LogLevel.DEBUG)
                    # Filter out all possible candidates for the current symbol
                    candidate_executors = [
                        ex
                        for ex in self.executors
                        if can_executor_execute(ex, current_bsym) and not isinstance(ex, FusionExecutor)
                    ]

                    if current_bsym.sym.id == PrimIDs.RETURN:
                        dict_time_strat["return"] = Executor(name=self.empty_executor_hashable_placeholder)
                        dict_mem_strat["return"] = Executor(name=self.empty_executor_hashable_placeholder)
                        # Add the modified return statement at the end of the for loop
                        break

                    # Not executors available
                    if not candidate_executors:
                        match_bsym_output(
                            current_bsym,
                            [dict_time_strat, dict_mem_strat],
                            Executor(name=self.empty_executor_hashable_placeholder),
                        )
                        continue
                    else:
                        log(f"Available executors for single region:\n{candidate_executors}", level=LogLevel.DEBUG)

                    # Helpers
                    candidate_best_time = BenchmarkResult()
                    candidate_best_mem = BenchmarkResult()
                    # Search for best candidate, by default remat will be called to find the optimal choice
                    # TODO: enable requests for no remat becnhmarks
                    # TODO: we should consider also FusionExecutor that can execute this single bsym in this beam search
                    for i, candidate in enumerate(candidate_executors):
                        # Match the current candidate to benchmark partial trace
                        match_bsym_output(current_bsym, [dict_time_strat, dict_mem_strat], candidate)
                        # Retrieve partial trace and benchmark, apply remat if possible
                        trc, _, _ = get_placed_trace(dict_time_strat, increasing_symbols)
                        # Apply fw bw remat
                        # if self.trace_type == TraceType.BW and self.active_fw_trace_ctx[0] is not None:
                        #     _, trc = rematerialize_forward_and_backward(self.active_fw_trace_ctx[0], trc)
                        # Now, benchmark
                        t, m, _ = benchmark_trace(trc, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0])
                        # Update results
                        if t < candidate_best_time.runtime:
                            candidate_best_time = BenchmarkResult(time=t, index=i)
                        if m < candidate_best_mem.memory:
                            candidate_best_mem = BenchmarkResult(memory=m, index=i)

                    if candidate_best_time.index == -1 or candidate_best_mem.index == -1:
                        raise AssertionError(
                            f"Failed to get optimal single trace region candidate. Available candidates for {current_bsym.sym.name}:\n{candidate_executors}"
                        )

                    log(
                        f"Best time OperatorExecutor for single {current_bsym.sym.name}: {candidate_executors[candidate_best_time.index].name}",
                        level=LogLevel.DEBUG,
                    )
                    log(
                        f"Best mem OperatorExecutor for single {current_bsym.sym.name}: {candidate_executors[candidate_best_mem.index].name}",
                        level=LogLevel.DEBUG,
                    )

                    match_bsym_output(current_bsym, [dict_time_strat], candidate_executors[candidate_best_time.index])
                    match_bsym_output(current_bsym, [dict_mem_strat], candidate_executors[candidate_best_mem.index])
                    # Go to next bsym group
                    continue

                # Inside groups we should have alwasy tensors as out
                best_res_time = BenchmarkResult()
                best_res_mem = BenchmarkResult()

                # TODO (matteochen): Aggregate them
                best_placement_time = None
                best_keys_time = None
                best_placement_mem = None
                best_keys_mem = None

                def measure_and_update_result():
                    nonlocal best_res_time
                    nonlocal best_placement_time
                    nonlocal best_keys_time
                    nonlocal best_res_mem
                    nonlocal best_placement_mem
                    nonlocal best_keys_mem
                    trc, keys, placements = get_placed_trace(dict_time_strat, increasing_symbols)
                    # if self.trace_type == TraceType.BW and self.active_fw_trace_ctx[0] is not None:
                    #     _, trc = rematerialize_forward_and_backward(self.active_fw_trace_ctx[0], trc)
                    cost, mem, _ = benchmark_trace(trc, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0])
                    log(f"Placed trace (cost = {cost} ms, mem = {mem/(2**30)} GB)\n{trc}", level=LogLevel.DEBUG)
                    if cost < best_res_time.runtime or (cost == best_res_time.runtime and mem < best_res_time.memory):
                        best_res_time = BenchmarkResult(time=cost, memory=mem, trace=trc)
                        best_placement_time = placements
                        best_keys_time = keys
                    if mem < best_res_mem.memory or (mem == best_res_mem.memory and cost < best_res_mem.runtime):
                        best_res_mem = BenchmarkResult(time=cost, memory=mem, trace=trc)
                        best_placement_mem = placements
                        best_keys_mem = keys

                start_idx = 0
                # This is to accomodate the following TODO
                # TODO: investigate why <prims.embedding_backward> is failing with torchcompile if left alone
                if ex.name == "torchcompile":
                    last_embedding_idx = -1
                    for idx in range(0, len(group)):
                        if group[idx].sym.name == "embedding_backward":
                            last_embedding_idx = idx
                    log(f"last embedding idx: {last_embedding_idx}", level=LogLevel.DEBUG)
                    if last_embedding_idx != -1:
                        # Until last_embedding_idx (included) assigned to current fusion ex
                        for i in range(0, last_embedding_idx + 1, 1):
                            match_bsym_output(group[i], [dict_time_strat, dict_mem_strat], ex)

                        if last_embedding_idx == len(group) - 1:
                            # Benchmark
                            measure_and_update_result()

                        start_idx = last_embedding_idx + 1

                n_missing_bsyms = len(group) - start_idx
                # TODO (matteochen): consider to add the iteration with no fusion regions
                for i in range(0, n_missing_bsyms, n_missing_bsyms-1 if self.trace_type == TraceType.BW else 1):
                    # for i in range(0, n_missing_bsyms):
                    # From top to bottom (this will include the whole region)
                    # -> First iteration is the one with fusion region with single element
                    # -> Last iteration gives the complete fusion region
                    for j in range(start_idx, start_idx + i + 1, increment_factor):
                        match_bsym_output(group[j], [dict_time_strat, dict_mem_strat], ex)
                    for k in range(start_idx + i + 1, len(group), increment_factor):
                        match_bsym_output(
                            group[k],
                            [dict_time_strat, dict_mem_strat],
                            get_first_available_operator_executor(
                                bsym=group[k],
                                executors=self.executors,
                                empty_hash=self.empty_executor_hashable_placeholder,
                            ),
                        )
                    # Benchmark
                    measure_and_update_result()

                    # TODO (matteochen): consider if this can increase placement
                    # From bottom to up (this will exclude the full region as being handled in the for cycle above)
                    # -> First iteration is the one with len(fusion_region) - 1
                    # -> Last iteration gives no fusion regions
                    # for j in range(start_idx, start_idx + i + 1, increment_factor):
                    #     match_bsym_output(
                    #         group[j],
                    #         [dict_time_strat, dict_mem_strat],
                    #         get_first_available_operator_executor(
                    #             bsym=group[j],
                    #             executors=self.executors,
                    #             empty_hash=self.empty_executor_hashable_placeholder,
                    #         ),
                    #     )
                    # for k in range(start_idx + i + 1, len(group), increment_factor):
                    #     match_bsym_output(group[k], [dict_time_strat, dict_mem_strat], ex)

                    # # Benchmark this placement
                    # measure_and_update_result()

                if best_placement_time is None or best_keys_time is None:
                    raise AssertionError("Failed to get best time placement")
                if best_placement_mem is None or best_keys_mem is None:
                    raise AssertionError("Failed to get best placement")

                log(
                    f"For group {group_id} best placement with time cost = {best_res_time.runtime} ms:\n{best_res_time.trace}",
                    level=LogLevel.DEBUG,
                )
                log(
                    f"For group {group_id} best placement with mem cost = {best_res_mem.memory / (2**30)} GB:\n{best_res_mem.trace}",
                    level=LogLevel.DEBUG,
                )

                # Update our dict
                for n, p in zip(best_keys_time, best_placement_time):
                    dict_time_strat |= {n: p}
                # Update our dict
                for n, p in zip(best_keys_mem, best_placement_mem):
                    dict_mem_strat |= {n: p}

            # Generate the placement
            executors_time = []
            executors_mem = []
            for bsym in self.trace.bound_symbols:
                if bsym.sym.id == PrimIDs.RETURN:
                    if "return" not in dict_time_strat or "return" not in dict_mem_strat:
                        raise AssertionError(f"Expected key return in mapping {dict_time_strat} and {dict_mem_strat}")
                    executors_time.append(dict_time_strat["return"])
                    executors_mem.append(dict_mem_strat["return"])
                elif isinstance(bsym.output, Sequence):
                    seq_hash = sequence_hash(bsym.output)
                    if seq_hash not in dict_time_strat or seq_hash not in dict_mem_strat:
                        raise AssertionError(
                            f"Expected key {seq_hash} in mapping {dict_time_strat} and {dict_mem_strat}"
                        )
                    executors_time.append(dict_time_strat[seq_hash])
                    executors_mem.append(dict_mem_strat[seq_hash])
                elif (
                    isinstance(bsym.output, CollectionProxy)
                    or isinstance(bsym.output, TensorProxy)
                    or isinstance(bsym.output, IntegerProxy)
                    or isinstance(bsym.output, FloatProxy)
                ):
                    if bsym.output.name not in dict_time_strat or bsym.output.name not in dict_mem_strat:
                        raise AssertionError(
                            f"Expected key {bsym.output.name} in mapping {dict_time_strat} and {dict_mem_strat}"
                        )
                    executors_time.append(dict_time_strat[bsym.output.name])
                    executors_mem.append(dict_mem_strat[bsym.output.name])
                else:
                    raise AssertionError(f"Type not handled: {type(bsym.output)}")

            # For the forward trace we benchmark (memory) the mocked return statement as we don't know which
            # tensor will be returned after the rematerialize_forward_and_backward() call in order to do not underestimate the memory consumption
            trace = self.trace
            if self.trace_type == TraceType.FW:
                trace = from_trace(self.trace)
                trace.bound_symbols = list(self.trace.bound_symbols)
                trace.bound_symbols.pop()
                trace.bound_symbols.append(
                    self.trace.bound_symbols[-1].from_bsym(args=get_not_used_intermediate_outsputs(trace))
                )
            # Save the optimal traces (both for runtime and memory consumption) that we have found
            for executors, container in zip(
                [executors_mem, executors_time],
                [
                    self.fusion_strat_helper.optimized_traces_mem_benchmark_only,
                    self.fusion_strat_helper.optimized_traces_time_benchmark_only,
                ],
            ):
                trc = assign_executors(
                    in_trace=trace,
                    executor_list=executors,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                )
                # print(f"Assigned trace:\n{trc}")
                # if self.trace_type == TraceType.BW:
                #     # pass
                #     _, trc = rematerialize_forward_and_backward(self.active_fw_trace_ctx[0], trc)
                container.append({ex.name: trc})

            # Save executors in order to generate real fw and bw trace with correct output with the placer
            # We add any provided compile option reference
            self.executor_placement_options.placement_options_time.append(
                FusionExecutorsPlacementCtx(placement=executors_time, compile_options=executor_compile_option)
            )
            self.executor_placement_options.placement_options_mem.append(
                FusionExecutorsPlacementCtx(placement=executors_mem, compile_options=executor_compile_option)
            )

        # If executor specific compile option is activated we need to know where a specific
        # trace does come from and the zip logic afterward can not be employed with self.fusion_executors list
        self.fusion_executors_saved_for_later = []
        ex: FusionExecutor
        for ex in self.fusion_executors:
            if ex.name not in self.fusion_strat_helper.supported_executors:
                # log(f"Fusion operator not supported: {ex.name}. Skipping it.")
                continue

            log(f"Searching best placement for fusion executor = {ex.name}", level=LogLevel.INFO)

            # We try to enable fusion specific compile options only for fw traces
            # Backward traces will follow fw traces options
            ex_compile_opts = (
                self.known_fusion_ex_compile_options.get(ex.name, []) if self.trace_type == TraceType.FW else []
            )
            self.fusion_executors_saved_for_later.append(ex)

            # Always search with option disabled -> standard flow
            _search(ex)

            # Currently we are enabling one compile option at the time as testing all the permutations might need too much time.
            # TODO: Consider implementing patterns based on the executor under investingation
            if ex_compile_opts:
                log(f'{ex.name} compile options: {[option.fusion_tag for option in ex_compile_opts]}')
                for opt in ex_compile_opts:
                    # Search only if we have an instruction related to the compile option
                    op_in_trace: bool = operation_in_trace(trace=self.trace, op=opt.symbol_tag)
                    if op_in_trace:
                        self.fusion_executors_saved_for_later.append(ex)
                        wrap_fn_with_exeuctor_compile_option(opt, _search, ex, opt)

    """
    ################################################## Public methods ##################################################
    """

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        if not self.cached_fw_traces:
            raise AssertionError("Failed to obtain optimal fw traces")
        return [candidate.trace for candidate in self.cached_fw_traces]

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        return (
            (self.best_pair_runtime.fw, self.best_pair_runtime.bw)
            if self.optimizer_type == OptimizerType.RUNTIME
            else (self.best_pair_memory.fw, self.best_pair_memory.bw)
        )

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType):
        from thunder.core.transform_common import dce

        self.trace_type = trace_type
        # dce for the backward trace will be passed afterwards
        self.trace: TraceCtx = dce(trace) if trace_type == TraceType.FW else trace

        match self.trace_type:
            case TraceType.FW:
                log(
                    f"New forward trace to optimize (strat = {self.optimizer_type}):\n{self.trace}", level=LogLevel.INFO
                )
            # TODO (matteochen): support bw trace optimization even though with no fw traces cached (computational trace?)
            case TraceType.BW:
                if not self.cached_fw_traces:
                    raise AssertionError("Can not optimize backward traces before forward traces")
                log(
                    f"New backward trace to optimize (strat = {self.optimizer_type}):\n{self.trace}",
                    level=LogLevel.INFO,
                )

    def optimize(self):
        from thunder.core.transform_common import dce
        from thunder.executors.torch_autograd import update_bw_from_forward_optimization
        from thunder.backend_optimizer.utils import assign_executors

        def _optimize():
            # Reset fusion helpers
            self.fusion_strat_helper = FusionStratHelper()
            # Reset helpers data structures
            self.executor_placement_options = ExecutorPlacementOptions()

            self._search_candidates()

            if len(self.executor_placement_options.placement_options_time) != len(
                self.fusion_executors_saved_for_later
            ):
                raise AssertionError(
                    f"Unexpected time placement options size: {len(self.executor_placement_options.placement_options_time)}. Expected: {len(self.fusion_executors_saved_for_later)}"
                )
            if len(self.executor_placement_options.placement_options_mem) != len(self.fusion_executors_saved_for_later):
                raise AssertionError(
                    f"Unexpected mem placement options size: {len(self.executor_placement_options.placement_options_mem)}. Expected: {len(self.fusion_executors_saved_for_later)}"
                )
            for placement_ctx, ex in zip(
                self.executor_placement_options.placement_options_time, self.fusion_executors_saved_for_later
            ):
                trc = assign_executors(
                    in_trace=self.trace,
                    executor_list=placement_ctx.placement,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                    compile_data=self.compile_data,
                    fusion_executor_compile_options_to_activate=placement_ctx.compile_options,
                )
                self.fusion_strat_helper.optimized_traces_time.append(
                    {ex.name: tuple([trc, placement_ctx.compile_options])}
                )
            for placement_ctx, ex in zip(
                self.executor_placement_options.placement_options_mem, self.fusion_executors_saved_for_later
            ):
                trc = assign_executors(
                    in_trace=self.trace,
                    executor_list=placement_ctx.placement,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                    compile_data=self.compile_data,
                    fusion_executor_compile_options_to_activate=placement_ctx.compile_options,
                )
                self.fusion_strat_helper.optimized_traces_mem.append(
                    {ex.name: tuple([trc, placement_ctx.compile_options])}
                )

            # Filter out the optimal candidates for the current serach iteration
            self._filter_candidates()

        match self.trace_type:
            case TraceType.FW:
                # Clear any previous results
                self.cached_fw_traces = []
                _optimize()
            # We have multiple cached optimized fw traces, find the best backward
            # TODO: make this prettier with a machine state for example
            case TraceType.BW:
                # Clear any previous results
                self.out_traces_candidates = []

                # Cached the bw trace as we need to modify the input trace during the loop
                cached_self_trace = from_trace(self.trace)
                cached_self_trace.bound_symbols = list(self.trace.bound_symbols)

                # Now we can generate backward solutions from the cached fw traces
                for fw_trace_candidate in self.cached_fw_traces:
                    log(f"Backward optimization with fw from {fw_trace_candidate.label}", level=LogLevel.INFO)
                    # Restore the original bw trace
                    self.trace = from_trace(cached_self_trace)
                    self.trace.bound_symbols = list(cached_self_trace.bound_symbols)
                    # Set the current active cached forward trace context
                    log(
                        f'Current fw cached ctx:\n{fw_trace_candidate.trace}\nOptions: {fw_trace_candidate.compile_opt.fusion_tag if fw_trace_candidate.compile_opt is not None else "None"}',
                        level=LogLevel.DEBUG
                    )
                    self.active_fw_trace_ctx = fw_trace_candidate.trace, fw_trace_candidate.compile_opt

                    log(f"Input bw trace:\n{self.trace}", level=LogLevel.DEBUG)

                    self.trace = update_bw_from_forward_optimization(fw=fw_trace_candidate.trace, bw=self.trace)

                    if self.apply_bucketing_bw_trace:
                        from thunder.distributed.transforms import FSDPCommBucketing

                        self.trace = FSDPCommBucketing.apply_bucketing_to_backward_trace(self.trace)

                    # Not called in the constructor for bw traces
                    self.trace = dce(self.trace)

                    # Enable any forward active compilation flag
                    if fw_trace_candidate.compile_opt:
                        wrap_fn_with_exeuctor_compile_option(fw_trace_candidate.compile_opt, _optimize)
                    else:
                        _optimize()

                self.best_pair_runtime, self.best_pair_memory = self._best_runtime_and_memory_candidates(
                    self.out_traces_candidates
                )


class BackendOptimizer:
    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log=True,
        apply_bucketing_bw_trace: bool,
        log_file_name="autotune_debug.log",
        visualizer: Visualizer | None = None,
        optimizer_type: OptimizerType = OptimizerType.RUNTIME,
        optimizer_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BEST_FUSER,
        compile_data,
    ) -> None:
        if optimizer_algorithm != OptimizationAlgorithm.BEST_FUSER:
            raise AssertionError(f'Optimization {optimizer_algorithm} not implemented')
        self.optimizer: PlacerBase = (
            FusionPlacer_BeamSearch(
                priority_executors=priority_executors,
                produce_log=produce_log,
                apply_bucketing_bw_trace=apply_bucketing_bw_trace,
                log_file_name=log_file_name,
                visualizer=visualizer,
                optimizer_type=optimizer_type,
                compile_data=compile_data,
            )
        )

        log("Executors:", level=LogLevel.INFO)
        for e in priority_executors:
            log(
                f"{e.name} -> is operator = {isinstance(e, OperatorExecutor)}, is fusion = {isinstance(e, FusionExecutor)}",
                level=LogLevel.INFO,
            )

    def optimize(self):
        self.optimizer.optimize()

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType):
        self.optimizer.attach_trace(trace=trace, trace_type=trace_type)

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        return self.optimizer.get_optimal_fw_traces()

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        return self.optimizer.get_optimal_fw_bw_traces()

