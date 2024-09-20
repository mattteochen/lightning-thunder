from collections.abc import Callable, Sequence
from enum import Enum
from thunder.backend_optimizer.utils import (
    dump_traces_placement,
    map_executors_from_reduced_trace_to_complete_trace,
    operation_in_trace,
    wrap_fn_with_exeuctor_compile_option,
    apply_results_from_file,
)
from thunder.core.compile_data import get_compile_data
from thunder.core.prims import PrimIDs
from thunder.core.proxies import CollectionProxy, FloatProxy, IntegerProxy, TensorProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceCtx
from thunder.core.transforms import construct_trace
from thunder.extend import Executor, FusionExecutor, get_always_executors
from typing import Hashable
from thunder.backend_optimizer.utils import benchmark_trace, BenchmarkResult, OptimizerType, TraceType
import logging

logging.basicConfig(level=logging.INFO, format="[{name}]: {message}", style="{")
logger = logging.getLogger("Thunder Autotuner")

# Control if single trace regions or partial traces are benchmarked during OperatorExecutor tuning
_benchmark_single_trace_region = False


class OptimizationAlgorithm(Enum):
    """
    Represents the optimization technique used by the autotuner.
    """

    BEST_FUSER = 0


class FusionCompileOptionsHelper:
    """
    Represents compile options for a fusion executor.

    Attributes:
        fusion_tag (str): A label representing the fusion ops regarding a compile option (e.g. nv_linear).
        symbol_tag (str): The symbol name
        id (PrimIDs): The symbol id.
        impl (Callable): A callable implementation.
        checker (Callable): A callable checker.
    """

    def __init__(self, fusion_tag: str, symbol_tag: str, id: PrimIDs, impl: Callable, checker: Callable) -> None:
        self.fusion_tag: str = fusion_tag
        self.symbol_tag: str = symbol_tag
        self.id: PrimIDs = id
        self.impl: Callable = impl
        self.checker: Callable = checker


class FusionExecutorsPlacementCtx:
    """
    Represents a executor placement context.

    Attributes:
        placement (list): A list of executors.
        compile_options (FusionExecutorsPlacementCtx | None): Any compile options being used for the fusion executor contained in the placement.
    """

    def __init__(self, *, placement: list, compile_options: FusionCompileOptionsHelper | None = None) -> None:
        self.placement: list = placement
        self.compile_options: FusionCompileOptionsHelper | None = compile_options


class TraceCandidate:
    """
    Represents an optimal trace candidate.

    Attributes:
        trace (TraceCtx): The candidate trace.
        ctx (FusionExecutorsPlacementCtx): Trace's placement context.
        label (str): A generic label to identify this candidate.
    """

    def __init__(
        self,
        *,
        trace: TraceCtx,
        ctx: FusionExecutorsPlacementCtx,
        label: str,
    ) -> None:
        self.trace: TraceCtx = trace
        self.ctx: FusionExecutorsPlacementCtx = ctx
        self.label: str = label


class TraceCandidates:
    """
    Represents an optimal pair of trace candidates (compute time and memory consumption).

    Attributes:
        best_time (TraceCtx): The trace with the optimal runtime.
        best_mem (TraceCtx): The trace with the optimal peak memory consumption.
        placement_ctx_time (FusionExecutorsPlacementCtx): Trace placement context with exeuctors and any applied fusion compile options.
        placement_ctx_mem (FusionExecutorsPlacementCtx): Trace placement context with exeuctors and any applied fusion compile options.
    """

    def __init__(
        self,
        best_time: TraceCtx | None = None,
        best_mem: TraceCtx | None = None,
        placement_ctx_time: FusionExecutorsPlacementCtx | None = None,
        placement_ctx_mem: FusionExecutorsPlacementCtx | None = None,
    ) -> None:
        self.best_time: TraceCtx | None = best_time
        self.best_mem: TraceCtx | None = best_mem
        self.placement_ctx_time: FusionExecutorsPlacementCtx | None = placement_ctx_time
        self.placement_ctx_mem: FusionExecutorsPlacementCtx | None = placement_ctx_mem

    def __repr__(self) -> str:
        """
        Give a representation for the current object.

        Returns:
            str: A string as the representation of the current object
        """
        return f"\nBest runtime candidate:\n{self.best_time}\nBest memory candidate:\n{self.best_mem}"

    def is_set(self) -> bool:
        """
        Check that the optimal trace pair has been set.

        Returns:
            bool: A flag indicating if the optimal trace is not None.
        """
        return False if self.best_time is None or self.best_mem is None else True

    def attach_best_time_candidate(self, trace: TraceCtx, ctx: FusionExecutorsPlacementCtx | None = None):
        """
        Attach a new best time trace result.

        Args:
            trace (TraceCtx): The trace to assign.
            ctx (FusionExecutorsPlacementCtx | None): The trace placement context.
        """
        self.best_time = trace
        self.placement_ctx_time = ctx

    def attach_best_mem_candidate(self, trace: TraceCtx, ctx: FusionExecutorsPlacementCtx | None = None):
        """
        Attach a new best memory trace result.

        Args:
            trace (TraceCtx): The trace to assign.
            ctx (FusionExecutorsPlacementCtx | None): The trace placement context.
        """
        self.best_mem = trace
        self.placement_ctx_mem = ctx

    def iterable(self) -> tuple[tuple, tuple]:
        """
        Returns an iterable object over the traces paired with their contexts.

        Returns:
            tuple: A tuple with paired values of performance metric and its context.
        """
        return (self.best_time, self.placement_ctx_time), (self.best_mem, self.placement_ctx_mem)

    def trace_ctx_iterable(self) -> tuple[TraceCtx | None, TraceCtx | None]:
        """
        Returns an iterable object over the traces.

        Returns:
            tuple: A tuple of traces with time and memory consumption targets.
        """
        return self.best_time, self.best_mem

    def placement_ctx_iterable(self) -> tuple[FusionExecutorsPlacementCtx | None, FusionExecutorsPlacementCtx | None]:
        """
        Returns an iterable object over the placement contexts.

        Returns:
            tuple: A tuple of contexes referring to traces targetting compute time and peak memory consumption.
        """
        return self.placement_ctx_time, self.placement_ctx_mem


class OutputCandidate:
    """
    Represents a final output candidate: forward and backward trace pair.

    Attributes:
        fw (TraceCtx): The forward trace.
        bw (TraceCtx): The backward trace.
        executors_fw (list): The forward trace regions' executors
        executors_bw (list): The backward trace regions' executors
        compile_opt (FusionExecutorsPlacementCtx | None): Any compile options being used for a fusion executor in the forward trace.
        tot_cost (float): The total cost to execute the pair (ms for a time strategy and GB for a memory strategy).
        apply_remat (bool): If rematerialization has been applied.
    """

    def __init__(
        self,
        *,
        fw: TraceCtx,
        bw: TraceCtx,
        executors_fw: list[Executor],
        executors_bw: list[Executor],
        compile_opt: FusionCompileOptionsHelper | None = None,
        cost: float = 0.0,
        apply_remat: bool = False,
    ) -> None:
        self.fw: TraceCtx = fw
        self.bw: TraceCtx = bw
        self.executors_fw: list[Executor] = executors_fw
        self.executors_bw: list[Executor] = executors_bw
        self.compile_opt: FusionCompileOptionsHelper | None = compile_opt
        self.tot_cost: float = cost
        self.apply_remat: bool = apply_remat

    def __repr__(self) -> str:
        """
        Give a representation of the current object.

        Returns:
            str: A string representing the current object.
        """
        return f"Final output candidate: forward trace:\n{self.fw.__repr__()}\nFinal output candidate: backward trace:\n{self.bw.__repr__()}"


class FusionStratHelper:
    """
    Represents a helper structure for the fusion strategy.

    Attributes:
        supported_executors (set): A list of supported fusion executors.
        optimized_traces_mem (list): a list of dictionaries containing informations regarding the optimized traces for peak memory consumption.
        optimized_traces_mem_benchmark_only (list): a list of dictionaries containing informations regarding the optimized traces for peak memory consumption (used only for internal benchmarking).
        optimized_traces_time (list): a list of dictionaries containing informations regarding the optimized traces for total compute time.
        optimized_traces_time_benchmark_only (list): a list of dictionaries containing informations regarding the optimized traces for total compute time (used only for internal benchmarking).
    """

    def __init__(self) -> None:
        self.supported_executors: set = set(["nvfuser", "torchcompile"])
        self.optimized_traces_mem: list[dict[str | Hashable, tuple[TraceCtx, FusionExecutorsPlacementCtx | None]]] = []
        self.optimized_traces_mem_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_time: list[dict[str | Hashable, tuple[TraceCtx, FusionExecutorsPlacementCtx | None]]] = []
        self.optimized_traces_time_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []


class ExecutorPlacementOptions:
    """
    Represents an aggregate placement options for executors combining those that targets peak memory consumption and those for total compute time.

    Attributes:
        placement_options_mem (list): A list of placement contexts.
        placement_options_time (list): A list of placement contexts.
    """

    def __init__(self) -> None:
        self.placement_options_mem: list[FusionExecutorsPlacementCtx] = []
        self.placement_options_time: list[FusionExecutorsPlacementCtx] = []


class PlacerBase:
    """
    Represents a base (interface) class for a placement class.

    Attributes:
        always_executors (tuple): A list of always present executors.
        empty_executor_hashable_placeholder (str): A label representing en empty executor.
        executors (Sequence): A list of executors to use.
        fusion_executors (Sequence): A list of fusion executors to use.
        fusion_executors_saved_for_later (Sequence): A helper list containing maybe repeated fusion executors.
        debug_msg (str): A dynamic filled log message.
        log_file_name (str): The output log file name if generated.
        produce_log (bool): A tuning parameter to control log file generation.
        optimizer_type (OptimizerType): The optimization target.
        active_fw_trace_ctx (tuple): An active forward trace set to optimize backward.
        cached_fw_traces (list): Cached optimized forward traces.
        cached_computational_trace (TraceCtx): Original computational trace
        cached_computational_backward_trace (TraceCtx): Original computational backward trace
        bw_trace_candidates (TraceCandidate): An instance of trace candidates.
        best_pair_runtime (OutputCandidate): A final trace pair targetting the compute time.
        best_pair_memory (OutputCandidate): A final trace pair targetting the peak memory consumption.
        apply_bucketing_bw_trace (bool): A distributed flag.
        benchmark_iters (int): Benchmark iteration steps.
        compile_data (Any): Thunder compilation data.
    """

    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log: bool = False,
        apply_bucketing_bw_trace: bool,
        log_file_name: str,
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
        self.log_file_name: str = log_file_name
        self.produce_log: bool = produce_log

        self.optimizer_type: OptimizerType = optimizer_type

        self.active_fw_trace_ctx: tuple[TraceCtx | None, FusionExecutorsPlacementCtx | None] = None, None
        self.cached_fw_traces: list[TraceCandidate] = []
        self.cached_computational_trace: TraceCtx = TraceCtx()
        self.cached_computational_backward_trace: TraceCtx = TraceCtx()
        self.bw_trace_candidates: TraceCandidates = TraceCandidates()
        self.out_traces_candidates: list[OutputCandidate] = []
        self.best_pair_runtime: OutputCandidate
        self.best_pair_memory: OutputCandidate

        self.apply_bucketing_bw_trace: bool = apply_bucketing_bw_trace

        self.benchmark_iters: int = 5

        self.compile_data = compile_data

    def optimize(self):
        """
        Optimize the executor placement for the current trace.
        """
        pass

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType, apply_dce=True):
        """
        Attach a new trace for executors optimization.

        Args:
            trace: The trace to attach.
            trace_type: Forward or backward trace refrence.
        """
        pass

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        """
        Retrive the optimal forward traces that the object has tuned.
        """
        return []

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        """
        Retrive the optimal forward and backward trace pair.
        """
        return (TraceCtx(), TraceCtx())


class FusionPlacer_BeamSearch(PlacerBase):
    """
    Represents a placer targetting the fusion regions.

    Attributes:
        fusion_strat_helper: A helper structures to save intermediate values.
        executor_placement_options: A helper structures to save different intemediate executor placement.
        is_reduced: A flag indicating if the current trace under optimization is a reduced version of a bigger trace (by common blocks reduction).
        cached_original_trace: A reference to the original trace if the optmization is performed on a reduced version.
    """

    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log: bool = False,
        apply_bucketing_bw_trace: bool,
        log_file_name: str,
        optimizer_type: OptimizerType = OptimizerType.RUNTIME,
        compile_data,
    ) -> None:
        super().__init__(
            priority_executors=priority_executors,
            produce_log=produce_log,
            apply_bucketing_bw_trace=apply_bucketing_bw_trace,
            log_file_name=log_file_name,
            optimizer_type=optimizer_type,
            compile_data=compile_data,
        )

        # Strat fusion
        self.fusion_strat_helper: FusionStratHelper = FusionStratHelper()
        self.executor_placement_options: ExecutorPlacementOptions = ExecutorPlacementOptions()

        # nvFuser compile options
        if compile_data.compile_options.get("autotune_enable_nvfuser_all", False):
            from thunder.executors.nvfuserex_impl import linear, _linear_check
            from thunder.executors.nvfuserex_impl import matmul, _matmul_check

            self.known_fusion_ex_compile_options: dict[str | Hashable, list[FusionCompileOptionsHelper]] = {
                "nvfuser": [
                    FusionCompileOptionsHelper("nv_enable_linear", "linear", PrimIDs.LINEAR, linear, _linear_check),
                    FusionCompileOptionsHelper("nv_enable_matmul", "matmul", PrimIDs.MATMUL, matmul, _matmul_check),
                ]
            }
        else:
            self.known_fusion_ex_compile_options: dict[str | Hashable, list[FusionCompileOptionsHelper]] = {
                "nvfuser": []
            }

        # Transformer based models optimization
        # For models based on layers of transformer blocks we can optimize the tuning by researching the best placement
        # on the model with a single layer and then mirror the configuration to the other layers.
        self.is_reduced: bool = False
        self.cached_original_trace: TraceCtx | None = None

    """
    ################################################## Internal methods ##################################################
    """

    def _best_runtime_and_memory_candidates(self, candidates: Sequence[OutputCandidate]):
        """
        Retrive the best compute time and peak memory consumption trace pairs.

        Args:
            candidates: A sequence of possible candidates.
        """
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
            pair_options: list[
                tuple[TraceCtx, TraceCtx, FusionCompileOptionsHelper | None, list[Executor], list[Executor], bool]
            ] = [
                (pair.fw, pair.bw, pair.compile_opt, pair.executors_fw, pair.executors_bw, False),
                (remat_fw, remat_bw, pair.compile_opt, pair.executors_fw, pair.executors_bw, True),
            ]
            # Select the best options
            for pair_option in pair_options:
                fw, bw, compile_opt, executors_fw, executors_bw, remat_applied = pair_option

                pair_cost_time = 0
                pair_cost_mem = 0
                t, m, _ = benchmark_trace(fw, iters=self.benchmark_iters)
                logger.debug(f"Pair fw time: {t} ms, mem: {m/(2**30)} GB")
                pair_cost_time = pair_cost_time + t
                pair_cost_mem = pair_cost_mem + m
                t, m, _ = benchmark_trace(bw, iters=self.benchmark_iters, fw_trace=fw)
                logger.debug(f"Pair bw time: {t} ms, mem: {m/(2**30)} GB")
                pair_cost_time = pair_cost_time + t
                pair_cost_mem = pair_cost_mem + m

                if pair_cost_time < min_value_time:
                    best_pair_runtime = OutputCandidate(
                        fw=fw,
                        bw=bw,
                        compile_opt=compile_opt,
                        executors_fw=executors_fw,
                        executors_bw=executors_bw,
                        cost=pair_cost_time,
                        apply_remat=remat_applied,
                    )
                    logger.debug(f"New best runtime pair (no remat):\n{best_pair_runtime}")
                    min_value_time = pair_cost_time

                if pair_cost_mem < min_value_mem:
                    best_pair_memory = OutputCandidate(
                        fw=fw,
                        bw=bw,
                        compile_opt=compile_opt,
                        executors_fw=executors_fw,
                        executors_bw=executors_bw,
                        cost=pair_cost_mem,
                        apply_remat=remat_applied,
                    )
                    logger.debug(f"New best memory pair (no remat):\n{best_pair_memory}")
                    min_value_mem = pair_cost_mem

        return best_pair_runtime, best_pair_memory

    def _filter_candidates(self):
        """
        Reduce the solutions count by comparing different options across different fusion executors.

        For forward traces all the options are cached.
        """
        self.debug_msg += "Traces benchmarks:\n\n"

        # We cache every optimized fw traces as they might impact differently on the bw trace
        # Number of fw traces to cached are: #fusion_executors * 2
        def fw_benchmark():
            # The optimizer builds the results in order following the self.fusion_executors list order
            pair_time: dict
            pair_mem: dict
            for pair_time, pair_mem in zip(
                self.fusion_strat_helper.optimized_traces_time, self.fusion_strat_helper.optimized_traces_mem
            ):
                placement_ctx_time: FusionExecutorsPlacementCtx
                placement_ctx_mem: FusionExecutorsPlacementCtx
                trc_time: TraceCtx
                trc_mem: TraceCtx
                trc_time, placement_ctx_time = list(pair_time.values())[0]
                trc_mem, placement_ctx_mem = list(pair_mem.values())[0]
                label = list(pair_time.keys())[0]
                # TODO (matteochen): remove the benchmark here as will done later on the bw pass
                c, m, _ = benchmark_trace(trc_time, self.benchmark_iters)
                self.debug_msg += (
                    f"Trace name = [{label}] - Target: TIME - Time = {c} ms - Mem = {m / (2**30)} GB\n{trc_time}\n\n"
                )
                c, m, _ = benchmark_trace(trc_mem, self.benchmark_iters)
                self.debug_msg += (
                    f"Trace name = [{label}] - Target: MEM - Mem = {m / (2**30)} GB - Time = {c} ms\n{trc_mem}\n\n"
                )
                # For forward trace we cache the best placement for both runtime and memory for the current Fusion executor (represented by label)
                for t, ctx in zip([trc_time, trc_mem], [placement_ctx_time, placement_ctx_mem]):
                    logger.info(
                        f"Caching fw candidate [compile option: {ctx.compile_options.fusion_tag if ctx.compile_options else 'None'}]"
                    )
                    self.cached_fw_traces.append(
                        TraceCandidate(
                            trace=t,
                            ctx=ctx,
                            label=(label + "_enabled_" + ctx.compile_options.fusion_tag)
                            if ctx.compile_options is not None
                            else label,
                        )
                    )
            self.cached_computational_trace = self.trace

        def bw_benchmark():
            time_result = BenchmarkResult()
            memory_result = BenchmarkResult()

            # Find best trace for runtime
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_time_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]
                trace_time, trace_mem, _ = benchmark_trace(
                    trace, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0]
                )
                self.debug_msg += f"Trace name = [{label}] - Target: TIME - Time = {trace_time} ms - Mem = {trace_mem / (2**30)} GB\n{trace}\n\n"
                if trace_time < time_result.runtime:
                    time_result = BenchmarkResult(time=trace_time, memory=trace_mem, trace=trace, label=label, index=i)

            # Find best trace for memory
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_mem_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]

                trace_time, trace_mem, _ = benchmark_trace(
                    trace, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0]
                )
                self.debug_msg += f"Trace name = [{label}] - Target: MEM - Mem = {trace_mem / (2**30)} GB - Time = {trace_time} ms\n{trace}\n\n"
                if trace_mem < memory_result.memory:
                    memory_result = BenchmarkResult(
                        time=trace_time, memory=trace_mem, trace=trace, label=label, index=i
                    )

            # Here we have to recover the traces without the pass through remat in order to be compliant
            # with thunder flow as we might have request for no remat.
            trc, placement_ctx = list(self.fusion_strat_helper.optimized_traces_time[time_result.index].values())[0]
            self.bw_trace_candidates.attach_best_time_candidate(trc, placement_ctx)
            trc, placement_ctx = list(self.fusion_strat_helper.optimized_traces_mem[memory_result.index].values())[0]
            self.bw_trace_candidates.attach_best_mem_candidate(trc, placement_ctx)

            # Now, finally build the pair fw and bw traces
            # The current fw trace is set by the caller and we take it as is. All current bw traces optimizations are made with the fw trace set by the caller.

            assert self.active_fw_trace_ctx[0] is not None and self.active_fw_trace_ctx[1] is not None

            for bw in self.bw_trace_candidates.iterable():
                self.out_traces_candidates.append(
                    OutputCandidate(
                        fw=self.active_fw_trace_ctx[0],
                        bw=bw[0],
                        executors_fw=self.active_fw_trace_ctx[1].placement,
                        executors_bw=bw[1].placement,
                        compile_opt=self.active_fw_trace_ctx[1].compile_options,
                    )
                )

            self.cached_computational_backward_trace = self.trace

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
        """
        For the current trace generate all the placement candidates.

        For each fusion executor the time-memory pair candidates will be generated and cached.
        If any compile options for an executor is available, it will be take under consideration.

        Args:
            increment_factor: An integer controlling the increment step during the fusion exclusion to speed up the compilation.
        """
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols
        from thunder.backend_optimizer.utils import (
            get_not_used_intermediate_outsputs,
            sequence_hash,
            can_executor_execute,
            get_first_available_operator_executor,
            assign_executors,
        )

        def get_placed_trace(mapping: dict[str, Executor], bound_symbols_in: Sequence[BoundSymbol]):
            """
            Generates a trace with the requested executors.

            Args:
                mapping: a dictionary pointing to the assigned executor for a trace region.
                bound_symbols_in: Input trace regions.
            """
            trc = from_trace(self.trace)
            trc.bound_symbols = list(bound_symbols_in)

            # For this partial trace we have to return all not used tensors otherwise the dce remove them
            tensors = get_not_used_intermediate_outsputs(trc)

            forced_return_bsym = self.trace.bound_symbols[-1].from_bsym(args=tensors)

            executor_configuration = []
            empty_executor = Executor(name=self.empty_executor_hashable_placeholder)
            keys = []
            for bsym in trc.bound_symbols:
                if bsym.sym.name == "return":
                    raise AssertionError("Return statement should not be here")
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
                executors_list=executor_configuration,
                always_executors=self.always_executors,
                empty_str=self.empty_executor_hashable_placeholder,
            )
            return placed_trace, keys, executor_configuration

        def _search(ex: FusionExecutor, executor_compile_option: FusionCompileOptionsHelper | None = None):
            """
            For the given executor search and cached the best placements.

            Args:
                ex: A fusion executor.
                executor_placement_options: Any compile option this executor might activate.
            """

            def _should_fuse_nvfuser(a: Node, b: Node):
                """
                Fusable fn definition for nvFuser.

                Args:
                    a: First node.
                    b: Second node.
                """

                def _can_fuse_node(n: Node):
                    # if already merged, then node can be fused
                    if len(n.group_bsyms) > 1:
                        return True
                    bsym: BoundSymbol = n.group_bsyms[0]
                    can_fuse: bool = ex.can_fuse(bsym)
                    cuda_in_or_out: bool = ex.has_cuda_input_or_output(bsym)
                    return can_fuse and cuda_in_or_out

                return _can_fuse_node(a) and _can_fuse_node(b)

            def _should_fuse_torchcompile(a: Node, b: Node):
                """
                Fusable fn definition for torch.compile.

                Args:
                    a: First node.
                    b: Second node.
                """

                def _can_fuse_node(n: Node):
                    if len(n.group_bsyms) > 1:
                        return True
                    bsym: BoundSymbol = n.group_bsyms[0]
                    return ex.can_fuse(bsym)

                return _can_fuse_node(a) and _can_fuse_node(b)

            def match_bsym_executor(bsym_in: BoundSymbol, dicts: list[dict], ex_in: Executor):
                """
                Match a bound symbol to its executor.

                Args:
                    bsym_in: The bound symbol to match.
                    dicts: The matrching destination.
                    ex_in: The executor to assign.
                """
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
                case "nvfuser":
                    merge_fn = _should_fuse_nvfuser
                case "torchcompile":
                    merge_fn = _should_fuse_torchcompile
            bound_symbol_groups = fuse_bound_symbols(self.trace, merge_fn)
            logger.debug(f"Number of Fusion groups = {len(bound_symbol_groups)}")

            # Print fusion groups if requested
            # for id, group in enumerate(bound_symbol_groups):
            #     log(f"Group id: {id}", level=LogLevel.DEBUG)
            #     for sub in group:
            #         log(f"{sub.sym.name} -> out: {sub.output}", level=LogLevel.DEBUG)
            #     if log_level == LogLevel.DEBUG and len(group) > 0:
            #         print("\n")

            dict_time_strat: dict[str, Executor] = {}
            dict_mem_strat: dict[str, Executor] = {}
            increasing_symbols = []
            # Tuning starting point: iterate over all the groups.
            for group_id, group in enumerate(bound_symbol_groups):
                logger.debug(f"Fusion group id: {group_id}")
                logger.debug(
                    f"Fusion group start = [{group[0].output.name if hasattr(group[0].output, 'name') else 'unknown'} = {group[0].sym.name}]"
                )
                logger.debug(
                    f"Fusion group end   = [{group[-1].output.name if hasattr(group[-1].output, 'name') else 'unknown'} = {group[-1].sym.name}]"
                )

                if group[0].sym.name != "return":
                    increasing_symbols += group

                # We assign to a Fusion executor only region with at least 2 elements. Otherwise let the best OperatorExecutor pick the symbol up
                if len(group) < 2:
                    current_bsym = group[0]
                    logger.debug(
                        f"--> Single group: [{current_bsym.output.name if hasattr(current_bsym.output, 'name') else 'unknown'} = {current_bsym.sym.name}]"
                    )
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
                        match_bsym_executor(
                            current_bsym,
                            [dict_time_strat, dict_mem_strat],
                            Executor(name=self.empty_executor_hashable_placeholder),
                        )
                        continue
                    else:
                        logger.debug(f"Available executors for single region:\n{candidate_executors}")

                    # Helpers
                    candidate_best_time = BenchmarkResult()
                    candidate_best_mem = BenchmarkResult()

                    # No choices
                    if len(candidate_executors) == 1:
                        candidate_best_time = BenchmarkResult(index=0)
                        candidate_best_mem = BenchmarkResult(index=0)
                    else:
                        if _benchmark_single_trace_region:
                            # Define the standalone trace in order to benchmark this symbol
                            subtrace = construct_trace()(current_bsym.sym, *current_bsym.args, **current_bsym.kwargs)

                        # Search for best candidate
                        for i, candidate in enumerate(candidate_executors):
                            if _benchmark_single_trace_region:
                                from thunder.common import transform_for_execution

                                subtrace_placed = transform_for_execution(subtrace, executors_list=[candidate])[-1]
                                logger.debug(f"Subtrace to benchmark single symbol:\n{subtrace_placed}")
                                t, m, _ = benchmark_trace(
                                    subtrace_placed, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0]
                                )
                            else:
                                # Match the current candidate into helper dicts to benchmark partial trace
                                match_bsym_executor(current_bsym, [dict_time_strat, dict_mem_strat], candidate)
                                # Retrieve partial trace and benchmark, apply remat if possible
                                trc, _, _ = get_placed_trace(dict_time_strat, increasing_symbols)
                                t, m, _ = benchmark_trace(
                                    trc, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0]
                                )
                            logger.info(
                                f"Operator excutor [{candidate.name}] candidate perf (is single trace region: {_benchmark_single_trace_region}): {t} ms {m/(2**30)} GB"
                            )
                            # Update results
                            if t < candidate_best_time.runtime:
                                candidate_best_time = BenchmarkResult(time=t, index=i)
                            if m < candidate_best_mem.memory:
                                candidate_best_mem = BenchmarkResult(memory=m, index=i)

                    if candidate_best_time.index == -1 or candidate_best_mem.index == -1:
                        raise AssertionError(
                            f"Failed to get optimal single trace region candidate. Available candidates for {current_bsym.sym.name}:\n{candidate_executors}"
                        )

                    logger.debug(
                        f"Best time OperatorExecutor for single {current_bsym.sym.name}: {candidate_executors[candidate_best_time.index].name}"
                    )
                    logger.debug(
                        f"Best mem OperatorExecutor for single {current_bsym.sym.name}: {candidate_executors[candidate_best_mem.index].name}"
                    )

                    match_bsym_executor(current_bsym, [dict_time_strat], candidate_executors[candidate_best_time.index])
                    match_bsym_executor(current_bsym, [dict_mem_strat], candidate_executors[candidate_best_mem.index])
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
                    cost, mem, _ = benchmark_trace(trc, self.benchmark_iters, fw_trace=self.active_fw_trace_ctx[0])
                    logger.debug(f"Placed trace (cost = {cost} ms, mem = {mem/(2**30)} GB)\n{trc}")
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
                    logger.debug(f"last embedding idx: {last_embedding_idx}")
                    if last_embedding_idx != -1:
                        # Until last_embedding_idx (included) assigned to current fusion ex
                        for i in range(0, last_embedding_idx + 1, 1):
                            match_bsym_executor(group[i], [dict_time_strat, dict_mem_strat], ex)

                        if last_embedding_idx == len(group) - 1:
                            # Benchmark
                            measure_and_update_result()

                        start_idx = last_embedding_idx + 1

                n_missing_bsyms = len(group) - start_idx
                # Tune a single fusion group.
                # NOTE: currently this is disabled for backward traces
                for i in range(0, n_missing_bsyms, n_missing_bsyms - 1 if self.trace_type == TraceType.BW else 1):
                    if ex.name == "torchcompile":
                        import torch

                        torch.compiler.reset()

                    # for i in range(0, n_missing_bsyms):
                    # From top to bottom (this will include the whole region)
                    # -> First iteration is the one with fusion region with single element
                    # -> Last iteration gives the complete fusion region
                    for j in range(start_idx, start_idx + i + 1, increment_factor):
                        match_bsym_executor(group[j], [dict_time_strat, dict_mem_strat], ex)
                    for k in range(start_idx + i + 1, len(group), increment_factor):
                        match_bsym_executor(
                            group[k],
                            [dict_time_strat, dict_mem_strat],
                            # In order to benchmark the fusion placecement, we can use any executor for the excluded bsym from the fusion region
                            # TODO: consider tuning the single trace regions removed from the fusion one
                            get_first_available_operator_executor(
                                bsym=group[k],
                                executors=self.executors,
                                empty_hash=self.empty_executor_hashable_placeholder,
                            ),
                        )
                    # Benchmark
                    measure_and_update_result()

                if best_placement_time is None or best_keys_time is None:
                    raise AssertionError("Failed to get best time placement")
                if best_placement_mem is None or best_keys_mem is None:
                    raise AssertionError("Failed to get best placement")

                logger.debug(
                    f"For group {group_id} best placement with time cost = {best_res_time.runtime} ms:\n{best_res_time.trace}"
                )
                logger.debug(
                    f"For group {group_id} best placement with mem cost = {best_res_mem.memory / (2**30)} GB:\n{best_res_mem.trace}"
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
            # tensor will be returned after the rematerialize_forward_and_backward call in order to do not under/over-estimate the memory consumption
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
                    executors_list=executors,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                )
                container.append({ex.name: trc})

            # We add any provided compile option reference
            self.executor_placement_options.placement_options_time.append(
                FusionExecutorsPlacementCtx(placement=executors_time, compile_options=executor_compile_option)
            )
            self.executor_placement_options.placement_options_mem.append(
                FusionExecutorsPlacementCtx(placement=executors_mem, compile_options=executor_compile_option)
            )

        # If any compile options is used we will need to have duplicated executors inside the executors list to maintain the matching.
        self.fusion_executors_saved_for_later = []
        ex: FusionExecutor
        for ex in self.fusion_executors:
            if ex.name not in self.fusion_strat_helper.supported_executors:
                continue

            logger.info(f"Searching best placement for fusion executor = {ex.name}")

            # We try to enable fusion specific compile options only for fw traces
            # Backward traces will follow fw traces options
            ex_compile_opts = (
                self.known_fusion_ex_compile_options.get(ex.name, []) if self.trace_type == TraceType.FW else []
            )
            self.fusion_executors_saved_for_later.append(ex)

            # Always search with option disabled (standard flow)
            _search(ex)

            # Currently we are enabling one compile option at the time as testing all the permutations might need too much time.
            # TODO: Consider implementing patterns based on the executor under investingation
            if ex_compile_opts:
                logger.info(f"{ex.name} compile options: {[option.fusion_tag for option in ex_compile_opts]}")
                for opt in ex_compile_opts:
                    # Search only if we have an instruction related to the compile option
                    op_in_trace: bool = operation_in_trace(trace=self.trace, op=opt.symbol_tag)
                    if op_in_trace:
                        self.fusion_executors_saved_for_later.append(ex)
                        wrap_fn_with_exeuctor_compile_option(opt, _search, ex, opt)

            logger.info(f"Searching best placement for fusion executor = {ex.name} ended.")

    """
    ################################################## Public methods ##################################################
    """

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        if not self.cached_fw_traces:
            raise AssertionError("Failed to obtain optimal fw traces")
        return [candidate.trace for candidate in self.cached_fw_traces]

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        restore_file = self.compile_data.compile_options.get("autotune_restore_configuration", "")

        # We apply the dce transform as it will be applied to the cached traces during the past optimization
        # (dce has been applied to the traces saved in the configuration).
        if restore_file:
            from thunder.core.transforms import dce

            fw_extrace, bw_extrace = apply_results_from_file(
                fw_trace=dce(self.cached_computational_trace),
                bw_trace=dce(self.cached_computational_backward_trace),
                file=restore_file,
            )
            return fw_extrace, bw_extrace
        return (
            (self.best_pair_runtime.fw, self.best_pair_runtime.bw)
            if self.optimizer_type == OptimizerType.RUNTIME
            else (self.best_pair_memory.fw, self.best_pair_memory.bw)
        )

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType, apply_dce: bool = True):
        from thunder.core.transform_common import dce

        self.trace_type = trace_type
        # dce for the backward trace will be passed afterwards as we might modify it before
        self.trace: TraceCtx = dce(trace) if apply_dce else trace

        match self.trace_type:
            case TraceType.FW:
                logger.info(f"New forward trace to optimize (strat = {self.optimizer_type})")
            # TODO (matteochen): support bw trace optimization even though with no fw traces cached (computational trace?)
            case TraceType.BW:
                if not self.compile_data.compile_options.get("autotune_restore_configuration", ""):
                    if not self.cached_fw_traces:
                        raise AssertionError("Can not optimize backward traces before forward traces")
                logger.info(f"New backward trace to optimize (strat = {self.optimizer_type})")

    def optimize(self):
        from thunder.core.transform_common import dce
        from thunder.executors.torch_autograd import update_bw_from_forward_optimization
        from thunder.backend_optimizer.utils import assign_executors
        from thunder.backend_optimizer.utils import repetead_trace_blocks, reduce_common_trace_blocks

        def _optimize():
            # Reset fusion helpers
            self.fusion_strat_helper = FusionStratHelper()
            # Reset helpers data structures
            self.executor_placement_options = ExecutorPlacementOptions()

            cd = get_compile_data()
            # Check if common blocks optimization is requested
            optimize_common_blocks = (
                False if cd is None else cd.compile_options.get("autotune_optimize_common_blocks", False)
            )
            optimize_common_blocks_min_size = (
                -1 if cd is None else cd.compile_options.get("autotune_optimize_common_blocks_min_size", -1)
            )

            # Cut the compilation time if possible
            common_trace_blocks = repetead_trace_blocks(
                trace=self.trace, min_block_size=optimize_common_blocks_min_size if optimize_common_blocks else -1
            )
            # A valid block is defined with at least 2 trace regions
            if len(common_trace_blocks) >= 2 and optimize_common_blocks:
                logger.info(
                    f"Running optimization with common blocks reduction. Found block indices in trace: {common_trace_blocks}"
                )
                reduced_trace = reduce_common_trace_blocks(trace=self.trace, common_blocks_in=common_trace_blocks)
                logger.info("Operating on reduced trace (by cutting common transformer blocks)")
                self.is_reduced = True
                self.cached_original_trace = self.trace
                self.trace = reduced_trace
            else:
                logger.info(
                    "Optimizing the whole trace directly. No common transformer block optimization will be applied."
                )

            # This performs executor tuning
            self._search_candidates()

            # From now on we have the optimized executors for each trace region. Apply them...
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

            # If we optimized the reduced trace we now can share the placing with other blocks
            if self.is_reduced and self.cached_original_trace is not None:
                for placement_ctx in self.executor_placement_options.placement_options_time:
                    placement = map_executors_from_reduced_trace_to_complete_trace(
                        self.cached_original_trace, common_trace_blocks, placement_ctx.placement
                    )
                    placement_ctx.placement = placement

                for placement_ctx in self.executor_placement_options.placement_options_mem:
                    placement = map_executors_from_reduced_trace_to_complete_trace(
                        self.cached_original_trace, common_trace_blocks, placement_ctx.placement
                    )
                    placement_ctx.placement = placement

                # Reset original trace
                self.trace = self.cached_original_trace
            # We will create the best compute time and peak memory consumption placement for each fusion executor
            for placement_ctx, ex in zip(
                self.executor_placement_options.placement_options_time, self.fusion_executors_saved_for_later
            ):
                trc = assign_executors(
                    in_trace=self.trace,
                    executors_list=placement_ctx.placement,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                    compile_data=self.compile_data,
                    fusion_executor_compile_options_to_activate=placement_ctx.compile_options,
                )
                self.fusion_strat_helper.optimized_traces_time.append({ex.name: tuple([trc, placement_ctx])})
            for placement_ctx, ex in zip(
                self.executor_placement_options.placement_options_mem, self.fusion_executors_saved_for_later
            ):
                trc = assign_executors(
                    in_trace=self.trace,
                    executors_list=placement_ctx.placement,
                    always_executors=self.always_executors,
                    empty_str=self.empty_executor_hashable_placeholder,
                    compile_data=self.compile_data,
                    fusion_executor_compile_options_to_activate=placement_ctx.compile_options,
                )
                self.fusion_strat_helper.optimized_traces_mem.append({ex.name: tuple([trc, placement_ctx])})

            # Filter out the optimal candidates for the current serach iteration
            self._filter_candidates()

        restore_file_name = self.compile_data.compile_options.get("autotune_restore_configuration", "")

        match self.trace_type:
            case TraceType.FW:
                # Perform optimization only if we don't restore it from a past configuration
                if restore_file_name:
                    self.cached_computational_trace = self.trace
                    logger.info("Skipping forward trace optimization as it will be restored from a configuration file.")
                    return

                # Clear any previous results
                self.cached_fw_traces = []
                _optimize()
            # We have multiple cached optimized fw traces, this iteration will create a fw-bw pair for
            # every cached forward trace. At the end the best one will be picked up.
            case TraceType.BW:
                # Perform optimization only if we don't restore it from a past configuration
                if restore_file_name:
                    logger.info(
                        "Skipping backward trace optimization as it will be restored from a configuration file."
                    )
                    self.cached_computational_backward_trace = self.trace
                    return

                # Clear any previous results
                self.out_traces_candidates = []

                # Cached the bw trace as we need to modify the self.trace during the loop
                cached_self_trace = from_trace(self.trace)
                cached_self_trace.bound_symbols = list(self.trace.bound_symbols)

                # Now we can generate backward solutions from the cached fw traces
                for fw_trace_candidate in self.cached_fw_traces:
                    logger.info(f"Backward optimization with fw from {fw_trace_candidate.label}")
                    # Restore the original bw trace
                    self.trace = from_trace(cached_self_trace)
                    self.trace.bound_symbols = list(cached_self_trace.bound_symbols)
                    # Set the current active cached forward trace context
                    # logger.info(
                    #     f"Current fw cached ctx:\n{fw_trace_candidate.trace}\nOptions: {fw_trace_candidate.ctx.compile_options.fusion_tag if fw_trace_candidate.ctx.compile_options is not None else 'None'}"
                    # )
                    self.active_fw_trace_ctx = fw_trace_candidate.trace, fw_trace_candidate.ctx

                    logger.debug(f"Input bw trace:\n{self.trace}")

                    self.trace = update_bw_from_forward_optimization(fw=fw_trace_candidate.trace, bw=self.trace)

                    # Taken from: https://github.com/Lightning-AI/lightning-thunder/blob/339a782e3d75061a065a3d2e47b5206f23aea7c3/thunder/executors/torch_autograd.py#L222
                    if self.apply_bucketing_bw_trace:
                        from thunder.distributed.transforms import FSDPCommBucketing

                        self.trace = FSDPCommBucketing.apply_bucketing_to_backward_trace(self.trace)

                    # Not called in the constructor for bw traces
                    self.trace = dce(self.trace)

                    # Enable any forward active compilation flag
                    if fw_trace_candidate.ctx.compile_options:
                        wrap_fn_with_exeuctor_compile_option(fw_trace_candidate.ctx.compile_options, _optimize)
                    else:
                        _optimize()

                # For every pair being generated filter out the best choice.
                self.best_pair_runtime, self.best_pair_memory = self._best_runtime_and_memory_candidates(
                    self.out_traces_candidates
                )

                # Save the tuning if requested
                do_save = self.compile_data.compile_options.get("autotune_save_configuration", False)
                if do_save:
                    model_name = self.compile_data.compile_options.get("model_name", "unknown")
                    file_name = f"{model_name}_runtime.json"
                    dump_traces_placement(
                        fw_trace=self.cached_computational_trace,
                        bw_trace=self.cached_computational_backward_trace,
                        file_name=file_name,
                        apply_remat=self.best_pair_runtime.apply_remat,
                        exs_fw=self.best_pair_runtime.executors_fw,
                        exs_bw=self.best_pair_runtime.executors_bw,
                    )
                    file_name = f"{model_name}_memory.json"
                    dump_traces_placement(
                        fw_trace=self.cached_computational_trace,
                        bw_trace=self.cached_computational_backward_trace,
                        file_name=file_name,
                        apply_remat=self.best_pair_memory.apply_remat,
                        exs_fw=self.best_pair_memory.executors_fw,
                        exs_bw=self.best_pair_memory.executors_bw,
                    )


class BackendOptimizer:
    """
    Represents a generic backend optimizer.

    Attributes:
        optimizer: An optimizer instance based on the configurations.
    """

    def __init__(
        self,
        *,
        priority_executors: Sequence[Executor],
        produce_log=False,
        apply_bucketing_bw_trace: bool,
        log_file_name="autotune_debug.log",
        optimizer_type: OptimizerType = OptimizerType.RUNTIME,
        optimizer_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BEST_FUSER,
        compile_data,
    ) -> None:
        if optimizer_algorithm != OptimizationAlgorithm.BEST_FUSER:
            raise AssertionError(f"Optimization {optimizer_algorithm} not implemented")
        self.optimizer: PlacerBase = FusionPlacer_BeamSearch(
            priority_executors=priority_executors,
            produce_log=produce_log,
            apply_bucketing_bw_trace=apply_bucketing_bw_trace,
            log_file_name=log_file_name,
            optimizer_type=optimizer_type,
            compile_data=compile_data,
        )

        logger.info(f"Executors: {[ex.name for ex in priority_executors]}")

    def optimize(self):
        """
        Optimize the executor placement for the current trace.
        """
        self.optimizer.optimize()

    def attach_trace(self, *, trace: TraceCtx, trace_type: TraceType, apply_dce=True):
        """
        Attach a new trace for executors optimization.

        Args:
            trace: The trace to attach.
            trace_type: Forward or backward trace refrence.
        """
        self.optimizer.attach_trace(trace=trace, trace_type=trace_type)

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        """
        Retrive the optimal forward traces that the object has tuned.
        """
        return self.optimizer.get_optimal_fw_traces()

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        """
        Retrive the optimal forward and backward trace pair.
        """
        return self.optimizer.get_optimal_fw_bw_traces()
