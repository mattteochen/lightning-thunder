from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain
from thunder.core.dtypes import dtype, is_boolean_dtype
from thunder.core.prims import PrimIDs
from thunder.core.utils import check, safe_map_flat
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.proxies import CollectionProxy, FloatProxy, IntegerProxy, Proxy, TensorProxy, variableify, Variable
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import from_trace, set_tracectx, reset_tracectx, get_tracectx, TraceCtx
from thunder.executors.data_dependent_partition import Node
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_always_executors
from thunder.visualizer.visualizer_helper import Visualizer
from typing import Any, Hashable
import thunder
import thunder.core.transforms as transforms
import torch

# Currently this manages both time and memory
class BenchmarkResult:
    def __init__(self) -> None:
        self.tm: float = float("inf")
        self.mem: float = float("inf")
        self.trace: TraceCtx | None = None
        self.label: str | Hashable = ""
        self.index = -1


class OptimizerType(Enum):
    MEMORY = 0
    RUNTIME = 1


class TraceType(Enum):
    FW = 0
    BW = 1


class OptimizationAlgorithm(Enum):
    BEST_FUSER = 0


class OptimizerNode:
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmark: float):
        self.candidate_executors[ex] = benchmark


class TraceCandidates:
    def __init__(self, best_time: TraceCtx | None = None, best_mem: TraceCtx | None = None) -> None:
        self.best_time: TraceCtx | None = best_time
        self.best_mem: TraceCtx | None = best_mem

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


class FinalOutputCandidates:
    def __init__(self, *, fw: TraceCtx, bw: TraceCtx, cost: float) -> None:
        self.fw: TraceCtx = fw
        self.bw: TraceCtx = bw
        self.tot_cost: float = cost

    def __repr__(self) -> str:
        return f"Final output candidate: forward trace:\n{self.fw.__repr__()}\nFinal output candidate: backward trace:{self.bw.__repr__()}"


# Benchmark only traces will contain traces after the rematerialization call with fw and bw calls, reproducing what will be the real traces after the autotune pass
# Non benchmark traces will contain traces after the placement (default) with no call to remat
# We have duplciated those in order to maintain thunder compilation flow as the output from the autotuner will be the traces with no pass through rematerialization
class FusionStratHelper:
    def __init__(self) -> None:
        self.supported_executors: set = set(["nvfuser", "torchcompile"])
        self.optimized_traces_mem: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_mem_benchmark_only: list[dict[str | Hashable, TraceCtx]] = [
        ]
        self.optimized_traces_time: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_time_benchmark_only: list[dict[str | Hashable, TraceCtx]] = [
        ]


class ExecutorPlacementOptions:
    def __init__(self) -> None:
        self.placement_options_mem: list[list[Executor]] = []
        self.placement_options_time: list[list[Executor]] = []


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1


log_level: LogLevel = LogLevel.INFO


def log(what: str, level: LogLevel):
    if log_level == LogLevel.DEBUG or log_level == level:
        print(
            f"================================================================================ Autotune: {what}")


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
    ) -> None:
        self.always_executors: tuple[Executor, ...] = get_always_executors()
        self.empty_executor_hashable_placeholder: str = "empty"
        self.executors: Sequence[Executor] = priority_executors
        self.fusion_executors: Sequence[FusionExecutor] = [
            ex for ex in self.executors if isinstance(ex, FusionExecutor)
        ]

        self.debug_msg: str = ""
        self.partial_costs: dict[TraceCtx, float] = {}
        self.visualizer: Visualizer | None = visualizer
        self.log_file_name: str = log_file_name
        self.produce_log: bool = produce_log

        self.optimizer_type: OptimizerType = optimizer_type
        self.optimization_algorithm: OptimizationAlgorithm | None = None

        self.active_fw_trace: TraceCtx | None = None
        self.cached_fw_traces: dict[str | Hashable, TraceCandidates] = {}
        self.bw_trace_candidates: TraceCandidates = TraceCandidates()
        self.out: list[FinalOutputCandidates] = []

        # Strat fusion
        self.fusion_strat_helper: FusionStratHelper = FusionStratHelper()
        self.executor_placement_options: ExecutorPlacementOptions = ExecutorPlacementOptions()

        self.apply_bucketing_bw_trace: bool = apply_bucketing_bw_trace

        self.benchmark_iters = 5

        log("Executors:", level=LogLevel.INFO)
        for e in self.executors:
            log(
                f"{e.name} -> is operator = {isinstance(e, OperatorExecutor)}, is fusion = {isinstance(e, FusionExecutor)}",
                level=LogLevel.INFO,
            )

    class SearchNode:
        def __init__(self, symbol: BoundSymbolInterface, idx: int) -> None:
            self.symbol = symbol
            self.idx = idx

    def attach_cached_fw_traces(self, cached_fw_traces: TraceCandidates, executor_name: str) -> None:
        self.cached_fw_traces[executor_name] = cached_fw_traces

    def attach_trace(self, *, trace: TraceCtx,  trace_type: TraceType):
        from thunder.core.transform_common import dce

        self.trace_type = trace_type
        # dce for the backward trace will be passed afterwards
        self.trace: TraceCtx = dce(
            trace) if trace_type == TraceType.FW else trace

        match self.trace_type:
            case TraceType.FW:
                log(
                    f"New forward trace to optimize (strat = {self.optimizer_type}):\n{self.trace}", level=LogLevel.INFO)
            # TODO (matteochen): support bw trace optimization even though with no fw traces cached
            case TraceType.BW:
                if not self.cached_fw_traces:
                    raise AssertionError(
                        "Can not optimize backward traces before forward traces")
                log(
                    f"New backward trace to optimize (strat = {self.optimizer_type}):\n{self.trace}", level=LogLevel.INFO)

    def place_optimizers(self, in_trace, executor_list: list[Executor]) -> TraceCtx:
        from thunder.executors.passes import _transform_for_operator_executor_execution

        swapmap: dict[Variable, Proxy] = {}

        # During the fusion pass and CSE optimizatons some args in trace regions could be different from the cached args. Restore the correct arguments
        # https://pytorch-lightning.slack.com/archives/C06QA9M8L3C/p1720732254341999
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
                            raise AssertionError(
                                f"What do you want to do here:\nobj_a:\n{obj_a}\nobj_b:{obj_b}")
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
            if ex.name == self.empty_executor_hashable_placeholder:
                return None

            execution_transform: None | Callable = ex.get_execution_transform(
                bsym.sym)
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
            raise AssertionError(
                "len(executor_list) != len(in_trace.bound_symbols)")

        # log(f'Visit transf')
        # for n, e in zip(in_trace.bound_symbols, executor_list):
        #     print(f'{n.sym.name} -> {e.name}')
        cached_subsymbols: dict[str, Sequence[BoundSymbol]] = {}
        executor_mapping: dict[str, Executor] = {}
        unique_fusion_executors = set()

        # Input should have equal length
        if len(executor_list) != len(in_trace.bound_symbols):
            raise AssertionError(
                "len(executor_list) != len(extrace.bound_symbols)")

        for b, e in zip(in_trace.bound_symbols, executor_list):
            if isinstance(e, FusionExecutor):
                unique_fusion_executors.add(e)
            if isinstance(b.output, TensorProxy):
                executor_mapping[b.output.name] = e

        extrace = transforms.visitor_transform_paired(
            in_trace, visit, zip(in_trace.bound_symbols, executor_list))

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
        extrace = _transform_for_operator_executor_execution(
            extrace, self.always_executors)

        return extrace

    def optimize(self, strat: OptimizationAlgorithm = OptimizationAlgorithm.BEST_FUSER):
        from thunder.core.transform_common import replace_redundant_inputs
        from thunder.core.transform_common import dce

        self.optimization_algorithm = strat

        def optmize_best_fuser():
            # Reset fusion helpers
            self.fusion_strat_helper = FusionStratHelper()
            # Reset helpers data structures
            self.executor_placement_options = ExecutorPlacementOptions()

            self.build_placement_options_best_fuser()

            if len(self.executor_placement_options.placement_options_time) != len(self.fusion_executors):
                raise AssertionError(
                    f"Unexpected time placement options size: {len(self.executor_placement_options.placement_options_time)}. Expected: {len(self.fusion_executors)}"
                )
            if len(self.executor_placement_options.placement_options_mem) != len(self.fusion_executors):
                raise AssertionError(
                    f"Unexpected mem placement options size: {len(self.executor_placement_options.placement_options_mem)}. Expected: {len(self.fusion_executors)}"
                )

            for placement, ex in zip(self.executor_placement_options.placement_options_time, self.fusion_executors):
                self.fusion_strat_helper.optimized_traces_time.append(
                    {ex.name: self.place_optimizers(self.trace, placement)}
                )
            for placement, ex in zip(self.executor_placement_options.placement_options_mem, self.fusion_executors):
                self.fusion_strat_helper.optimized_traces_mem.append(
                    {ex.name: self.place_optimizers(self.trace, placement)}
                )

            self.benchmark_traces()

        def match_optimizer_algorithm():
            match self.optimization_algorithm:
                case OptimizationAlgorithm.BEST_FUSER:
                    optmize_best_fuser()

        match self.trace_type:
            case TraceType.FW:
                match_optimizer_algorithm()
            # We have multiple cached optimized fw traces, find the best backward
            case TraceType.BW:
                # Cached the bw trace as we need to modify the input trace during the loop
                cached_self_trace = from_trace(self.trace)
                cached_self_trace.bound_symbols = list(
                    self.trace.bound_symbols)
                for label, candidate in self.cached_fw_traces.items():
                    log(f'Backward optimization with fw from {label}', level=LogLevel.INFO)
                    fw_traces = candidate.iterable()
                    for trc in fw_traces:

                        # TODO (matteochen): unify below with the original block

                        # Restore the original bw trace
                        self.trace = from_trace(cached_self_trace)
                        self.trace.bound_symbols = list(
                            cached_self_trace.bound_symbols)
                        # Set the current active cached forward trace
                        self.active_fw_trace = trc

                        log(f"Cached fw trace:\n{self.active_fw_trace}", level=LogLevel.DEBUG)
                        log(f"Input bw trace:\n{self.trace}", level=LogLevel.DEBUG)

                        # Some of the optimization passes change proxies in the trace and
                        # any change in the forward trace must be reflected in the backward
                        # trace.
                        original_bw_saved_tensors_for_backward = self.trace.args[0][0]
                        new_fw_saved_tensors_for_backward = trc.output[1][0]
                        swap_map = {
                            variableify(x): y
                            for x, y in zip(original_bw_saved_tensors_for_backward, new_fw_saved_tensors_for_backward)
                            if variableify(x) != variableify(y)
                        }
                        new_bsyms = replace_redundant_inputs(
                            swap_map, self.trace.bound_symbols)
                        # replace_redundant_inputs doesn't replace the output of
                        # UNPACK_SEQUENCE so we do it manually. Here we have certain
                        # assumptions about the structure of the backward trace.
                        assert self.trace.bound_symbols[0].sym.id == PrimIDs.UNPACK_TRIVIAL
                        assert self.trace.bound_symbols[0].kwargs["name"] == "saved_for_backward"
                        assert self.trace.bound_symbols[4].sym.id == PrimIDs.UNPACK_SEQUENCE
                        assert self.trace.bound_symbols[4].args[0].name == "C0"
                        new_bsyms[4] = new_bsyms[4].from_bsym_swap_proxies(
                            swap_map,
                            skip_inputs=False,
                            skip_output=False,
                            skip_subsymbols=False,
                        )
                        self.trace.bound_symbols = new_bsyms

                        if self.apply_bucketing_bw_trace:
                            from thunder.distributed.transforms import FSDPCommBucketing

                            self.trace = FSDPCommBucketing.apply_bucketing_to_backward_trace(
                                self.trace)

                        # Not called in the constructor for bw traces
                        dce(self.trace)

                        match_optimizer_algorithm()

    def can_executor_execute(self, ex: Executor, bsym: BoundSymbol) -> bool:
        try:
            return ex.can_execute(bsym)
        except:
            return False

    # For each fusion executor in the input list, find the best trace dispatching for each executor
    def build_placement_options_best_fuser(self, increment_factor: int = 1):
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols
        from thunder.core.rematerialization import rematerialize_forward_and_backward

        def sequence_hash(s: Sequence) -> str:
            name = ""
            for e in s:
                if (
                    isinstance(e, CollectionProxy)
                    or isinstance(e, TensorProxy)
                    or isinstance(e, IntegerProxy)
                    or isinstance(e, FloatProxy)
                ):
                    name += e.name
                # TODO (matteochen): investigate if this is suitable
                elif isinstance(e ,int):
                    name += f'int{e}'
                elif e is None:
                    name += "None"
                else:
                    raise AssertionError(
                        f"What? Maybe nested Sequence. type = {type(e)}")
            return name

        # TODO (matteochen): Benchmark the optimal executor and call this optimal
        def get_first_available_executor(bsym: BoundSymbol):
            for ex in self.executors:
                if isinstance(ex, FusionExecutor):
                    continue
                if self.can_executor_execute(ex, bsym):
                    return ex
            return Executor(name=self.empty_executor_hashable_placeholder)

        def get_placed_trace(mapping: dict[str, Executor], bound_symbols_in: Sequence[BoundSymbol]):
            # log(f'Input mapping len = {len(mapping)}:')
            # log(f'Input bound_symbols len = {len(bound_symbols_in)}:')
            trc = from_trace(self.trace)
            trc.bound_symbols = list(bound_symbols_in)

            # For this partial trace we have to return all not used tensors otherwise the dce will cut them out
            tensors = return_not_used_vars(trc)

            forced_return_bsym = self.trace.bound_symbols[-1].from_bsym(
                args=tensors)

            executor_configuration = []
            empty_executor = Executor(
                name=self.empty_executor_hashable_placeholder)
            keys = []
            for bsym in trc.bound_symbols:
                if bsym.sym.name == "return":
                    raise AssertionError("Return statement should not be here")
                    # executor_configuration.append(empty_executor)
                    # keys.append('return')
                elif isinstance(bsym.output, Sequence):
                    seq_hash = sequence_hash(bsym.output)
                    executor_configuration.append(
                        mapping.get(seq_hash, empty_executor))
                    keys.append(seq_hash)
                elif (
                    isinstance(bsym.output, CollectionProxy)
                    or isinstance(bsym.output, TensorProxy)
                    or isinstance(bsym.output, IntegerProxy)
                    or isinstance(bsym.output, FloatProxy)
                ):
                    if bsym.output.name not in mapping:
                        raise AssertionError(
                            f"Expected key {bsym.output.name} in mapping {mapping}")
                    executor_configuration.append(mapping[bsym.output.name])
                    keys.append(bsym.output.name)
                else:
                    raise AssertionError(
                        f"Type not handled: {type(bsym.output)}")

            if trc.bound_symbols[-1].sym.name != "return":
                trc.bound_symbols.append(forced_return_bsym)
                executor_configuration.append(
                    Executor(name=self.empty_executor_hashable_placeholder))
                keys.append("return")

            if len(trc.bound_symbols) != len(executor_configuration) or len(keys) != len(executor_configuration):
                raise AssertionError(
                    f"len trc.bound_symbols ({len(trc.bound_symbols)}) != len executor_configuration ({len(executor_configuration)}) != len keys ({len(keys)})"
                )

            # for b, e in zip(trc.bound_symbols, executor_configuration):
            #     if isinstance(b.output, TensorProxy):
            #         print(f'{b.sym.name}: {b.output.name} -> {e.name}')

            placed_trace = self.place_optimizers(trc, executor_configuration)
            return placed_trace, keys, executor_configuration

        ex: FusionExecutor
        for ex in self.fusion_executors:
            if ex.name not in self.fusion_strat_helper.supported_executors:
                raise AssertionError(
                    f"Fusion operator not supported: {ex.name}")

            log(
                f"Searching best placement for fusion executor = {ex.name}", level=LogLevel.DEBUG)

            # TODO (matteochen): each executor has a custom should fuse function, can we make this prettier?
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
                    raise AssertionError(
                        f"Type not handled: {type(bsym_in.output)}")

            bound_symbol_groups = fuse_bound_symbols(
                self.trace, _should_fuse_nvfuser if ex.name == "nvfuser" else _should_fuse_torchcompile
            )
            log(f"Num of groups = {len(bound_symbol_groups)}", level=LogLevel.DEBUG)

            for id, group in enumerate(bound_symbol_groups):
                log(f"Group id: {id}", level=LogLevel.DEBUG)
                for sub in group:
                    log(f"{sub.sym.name} -> out: {sub.output}", level=LogLevel.DEBUG)
                # if len(group) > 0:
                #     print("\n")

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
                    name = current_bsym.sym.name
                    # Filter out all possible candidates for the current symbol
                    candidate_executors = [ex for ex in self.executors if self.can_executor_execute(ex, current_bsym) and not isinstance(ex, FusionExecutor)]

                    if name == "return":
                        dict_time_strat["return"] = Executor(name=self.empty_executor_hashable_placeholder)
                        dict_mem_strat["return"] = Executor(name=self.empty_executor_hashable_placeholder)
                        # Add the modified return statement at the end of the for loop
                        break

                    # Not executors available
                    if not candidate_executors:
                        match_bsym_output(
                            current_bsym, [dict_time_strat, dict_mem_strat], Executor(name=self.empty_executor_hashable_placeholder))
                        continue
                    else:
                        log(f'Available executors for single region:\n{candidate_executors}', level=LogLevel.DEBUG)

                    # Helpers
                    candidate_best_time = BenchmarkResult()
                    candidate_best_mem = BenchmarkResult()
                    # Search for best candidate
                    for i, candidate in enumerate(candidate_executors):
                        # Match the current candidate to benchmark partial trace
                        match_bsym_output(
                            current_bsym, [dict_time_strat, dict_mem_strat], candidate)
                        # Retrieve partial trace and benchmark, apply remat if possible
                        trc, _, _ = get_placed_trace(
                            dict_time_strat, increasing_symbols)
                        if self.trace_type == TraceType.BW and self.active_fw_trace is not None:
                            _, trc = rematerialize_forward_and_backward(
                                self.active_fw_trace, trc)
                        t, m, _ = benchmark_trace(trc, self.benchmark_iters)
                        # Update results
                        if t < candidate_best_time.tm:
                            candidate_best_time.tm = t
                            candidate_best_time.index = i

                        if m < candidate_best_mem.mem:
                            candidate_best_mem.mem = m
                            candidate_best_mem.index = i

                    if candidate_best_time.index == -1 or candidate_best_mem.index == -1:
                        raise AssertionError(f'Failed to get optimal single trace region candidate. Available candidates for {name}:\n{candidate_executors}')

                    log(f'Best time OperatorExecutor for single {name}: {candidate_executors[candidate_best_time.index].name}', level=LogLevel.DEBUG)
                    log(f'Best mem OperatorExecutor for single {name}: {candidate_executors[candidate_best_mem.index].name}', level=LogLevel.DEBUG)

                    match_bsym_output(
                        current_bsym, [dict_time_strat], candidate_executors[candidate_best_time.index])
                    match_bsym_output(
                        current_bsym, [dict_mem_strat], candidate_executors[candidate_best_mem.index])
                    continue

                # Inside groups we should have alwasy tensors as out
                best_res_time = BenchmarkResult()
                best_res_mem = BenchmarkResult()
                worst_res_time = BenchmarkResult()
                worst_res_mem = BenchmarkResult()
                # Only for visual
                worst_res_mem.measure = 0
                worst_res_time.measure = 0

                # TODO (matteochen): Aggregate them
                best_placement_time = None
                best_keys_time = None
                best_placement_mem = None
                best_keys_mem = None

                def measure_and_update_result():
                    nonlocal best_res_time
                    nonlocal best_placement_time
                    nonlocal best_keys_time
                    nonlocal worst_res_time
                    nonlocal best_res_mem
                    nonlocal best_placement_mem
                    nonlocal best_keys_mem
                    nonlocal worst_res_mem
                    trc, keys, placements = get_placed_trace(
                        dict_time_strat, increasing_symbols)
                    if self.trace_type == TraceType.BW and self.active_fw_trace is not None:
                        _, trc = rematerialize_forward_and_backward(
                            self.active_fw_trace, trc)
                    cost, mem, out = benchmark_trace(trc, self.benchmark_iters)
                    del out
                    log(
                        f"Placed trace (cost = {cost} ms, mem = {mem/(2**30)} GB)\n{trc}", level=LogLevel.DEBUG)
                    if cost < best_res_time.tm or (cost == best_res_time.tm and mem < best_res_time.mem):
                        best_res_time.tm = cost
                        best_res_time.mem = mem
                        best_res_time.trace = trc
                        best_placement_time = placements
                        best_keys_time = keys
                    if cost > worst_res_time.tm:
                        worst_res_time.tm = cost

                    if mem < best_res_mem.mem or (mem == best_res_mem.mem and cost < best_res_mem.tm):
                        best_res_mem.tm = cost
                        best_res_mem.mem = mem
                        best_res_mem.trace = trc
                        best_placement_mem = placements
                        best_keys_mem = keys
                    if mem > worst_res_mem.mem:
                        worst_res_mem.mem = mem

                start_idx = 0
                # This is to accomodate the following TODO
                # TODO: investigate why <prims.embedding_backward> is failing with torchcompile if left alone
                if ex.name == "torchcompile":
                    last_embedding_idx = -1
                    for idx in range(0, len(group)):
                        if group[idx].sym.name == "embedding_backward":
                            last_embedding_idx = idx
                    log(f"last embedding {last_embedding_idx}", level=LogLevel.DEBUG)
                    if last_embedding_idx != -1:
                        # Until last_embedding_idx (included) assigned to current fusion ex
                        for i in range(0, last_embedding_idx + 1, 1):
                            match_bsym_output(
                                group[i], [dict_time_strat, dict_mem_strat], ex)

                        if last_embedding_idx == len(group) - 1:
                            # Benchmark
                            measure_and_update_result()

                        start_idx = last_embedding_idx + 1

                n_missing_bsyms = len(group) - start_idx
                for i in range(0, n_missing_bsyms, n_missing_bsyms-1 if self.trace_type == TraceType.BW else 1):
                # for i in range(0, n_missing_bsyms):
                    # From top to bottom (this will include the whole region)
                    # -> First iteration is the one with fusion region with single element
                    # -> Last iteration gives the complete fusion region

                    for j in range(start_idx, start_idx + i + 1, increment_factor):
                        match_bsym_output(
                            group[j], [dict_time_strat, dict_mem_strat], ex)
                    for k in range(start_idx + i + 1, len(group), increment_factor):
                        match_bsym_output(
                            group[k], [dict_time_strat, dict_mem_strat], get_first_available_executor(
                                group[k])
                        )
                    # Benchmark
                    measure_and_update_result()

                    # TODO (matteochen): consider if this can increase placement
                    # From bottom to up (this will exclude the full region as being handled in the for cycle above)
                    # -> First iteration is the one with len(fusion_region) - 1
                    # -> Last iteration gives no fusion regions
                    # for j in range(0, i+1, increment_factor):
                    #     dict_time_strat[group[j].output.name] = get_default_executor(group[j])
                    # for k in range(i+1, len(group), increment_factor):
                    #     dict_time_strat[group[k].output.name] = ex

                    # Benchmark this placement
                    # measure_and_update_result()

                if best_placement_time is None or best_keys_time is None:
                    raise AssertionError("Failed to get best time placement")
                if best_placement_mem is None or best_keys_mem is None:
                    raise AssertionError("Failed to get best placement")

                log(
                    f"For group {group_id} best placement with time cost = {best_res_time.tm} ms (worst time = {worst_res_time.tm} ms):\n{best_res_time.trace}",
                    level=LogLevel.DEBUG
                )
                log(
                    f"For group {group_id} best placement with mem cost = {best_res_mem.mem / (2**30)} GB (worst mem = {worst_res_mem.mem/(2**30)} GB) is:\n{best_res_mem.trace}",
                    level=LogLevel.DEBUG
                )

                # for n, p in zip(best_keys, best_placement):
                #     print(f'{n} -> {p.name}')

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
                if bsym.sym.name == "return":
                    if "return" not in dict_time_strat or "return" not in dict_mem_strat:
                        raise AssertionError(
                            f"Expected key return in mapping {dict_time_strat} and {dict_mem_strat}")
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
                    raise AssertionError(
                        f"Type not handled: {type(bsym.output)}")

            # For the forward trace we benchmark (memory) the mocked return statement as we don't know which
            # Tensor will be returned after the rematerialize_forward_and_backward() call in order to do not overestimate the memory consumption
            if self.trace_type == TraceType.FW:
                trc = from_trace(self.trace)
                trc.bound_symbols = list(self.trace.bound_symbols)
                trc.bound_symbols.pop()
                trc.bound_symbols.append(
                    self.trace.bound_symbols[-1].from_bsym(args=return_not_used_vars(trc)))
                # NOTE: Here the active trace to place will be 'trc' and not 'self.trace'
                trc_time = self.place_optimizers(trc, executors_mem)
                c, m, o = benchmark_trace(trc_time, self.benchmark_iters)
                del o
                log(f"Debug MEM, mem = {m/(2**30)} GB:\n{trc_time}", level=LogLevel.DEBUG)
                self.fusion_strat_helper.optimized_traces_mem_benchmark_only.append({
                                                                                    ex.name: trc_time})
                trc_mem = self.place_optimizers(trc, executors_time)
                c, m, o = benchmark_trace(trc_mem, self.benchmark_iters)
                del o
                log(f"Debug TIME, time = {c} ms:\n{trc_mem}", level=LogLevel.DEBUG)
                self.fusion_strat_helper.optimized_traces_time_benchmark_only.append({
                                                                                     ex.name: trc_mem})
            else:
                trc = self.place_optimizers(self.trace, executors_mem)
                _, trc = rematerialize_forward_and_backward(
                    self.active_fw_trace, trc)
                c, m, o = benchmark_trace(trc, self.benchmark_iters)
                del o
                log(f"Debug MEM, mem = {m/(2**30)} GB:\n{trc}", level=LogLevel.DEBUG)
                self.fusion_strat_helper.optimized_traces_mem_benchmark_only.append({
                                                                                    ex.name: trc})
                trc = self.place_optimizers(self.trace, executors_time)
                _, trc = rematerialize_forward_and_backward(
                    self.active_fw_trace, trc)
                c, m, o = benchmark_trace(trc, self.benchmark_iters)
                del o
                log(f"Debug TIME, time = {c} ms:\n{trc}", level=LogLevel.DEBUG)
                self.fusion_strat_helper.optimized_traces_time_benchmark_only.append({
                                                                                     ex.name: trc})

            # Save executors in order to generate real fw and bw trace with correct output
            self.executor_placement_options.placement_options_time.append(
                executors_time)
            self.executor_placement_options.placement_options_mem.append(
                executors_mem)

    def get_optimal_fw_traces(self) -> Sequence[TraceCtx]:
        if not self.cached_fw_traces:
            raise AssertionError("Failed to obtain optimal fw traces")
        return [getattr(candidate, field) for candidate in self.cached_fw_traces.values() for field in ['best_time', 'best_mem']]

    def get_optimal_fw_bw_traces(self) -> tuple[TraceCtx, TraceCtx]:
        # This is agnostic from the optimization strat as results are both floats
        min_value: float = float("inf")
        ans: FinalOutputCandidates | None = None
        log(f'Computing the best pair option (tot options = {len(self.out)})', level=LogLevel.INFO)
        for pair in self.out:
            if pair.tot_cost < min_value:
                log(f"New best pair:\n{pair}", level=LogLevel.INFO)
                min_value = pair.tot_cost
                ans = pair
        if ans is None:
            raise AssertionError('Best pair not found')
        return ans.fw, ans.bw

    def bsym_assigned(self, bsym: BoundSymbol) -> bool:
        return isinstance(bsym.sym.executor, OperatorExecutor) or isinstance(bsym.sym.executor, FusionExecutor)

    def benchmark_traces(self):

        self.debug_msg += "Traces benchmarks:\n\n"

        # We cached every optimized fw traces as they might impact differently on the bw trace
        # Number of fw traces to cached are: #fusion_executors * 2
        def fw_benchmark():
            match self.optimization_algorithm:
                case OptimizationAlgorithm.BEST_FUSER:
                    # The optimizator builds the results in order following the self.fusion_executors list order
                    for pair_time, pair_mem in zip(self.fusion_strat_helper.optimized_traces_time, self.fusion_strat_helper.optimized_traces_mem):
                        # pair is a dict
                        trc_time = list(pair_time.values())[0]
                        trc_mem = list(pair_mem.values())[0]
                        label = list(pair_time.keys())[0]
                        # TODO (matteochen): remove the benchmark here as will done later on the bw pass
                        c, m, _ = benchmark_trace(trc_time, self.benchmark_iters)
                        log(
                            f'Benchmark fw end: Trace = [{label}] (time = {c} ms, mem = {m / (2**30)} GB)":\n{trc_time}', level=LogLevel.INFO)
                        self.debug_msg += (
                                f"Trace name = [{label}] - Target: TIME - Time = {c} ms - Mem = {m / (2**30)} GB\n{trc_time}\n\n"
                        )
                        c, m, _ = benchmark_trace(trc_mem, self.benchmark_iters)
                        log(
                            f'Benchmark fw end: Trace = [{label}] (time = {c} ms, mem = {m / (2**30)} GB)":\n{trc_mem}', level=LogLevel.INFO)
                        self.debug_msg += (
                                f"Trace name = [{label}] - Target: MEM - Mem = {m / (2**30)} GB - Time = {c} ms\n{trc_mem}\n\n"
                        )

                        self.cached_fw_traces[label] = TraceCandidates(best_time = trc_time, best_mem = trc_mem)


        def bw_benchmark():
            time_result = BenchmarkResult()
            memory_result = BenchmarkResult()

            # Find best trace for runtime
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_time_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]
                trace_time, trace_mem, res = benchmark_trace(trace, self.benchmark_iters)
                self.debug_msg += (
                        f"Trace name = [{label}] - Target: TIME - Time = {trace_time} ms - Mem = {trace_mem / (2**30)} GB\n{trace}\n\n"
                )
                log(
                    f'Benchmark trace (target TIME) "{label}" (time = {trace_time} ms, mem = {trace_mem / (2**30)} GB:\n{trace}', level=LogLevel.INFO
                )
                if trace_time < time_result.tm:
                    time_result.tm = trace_time
                    time_result.mem = trace_mem
                    time_result.trace = trace
                    time_result.label = label
                    time_result.index = i

            # Find best trace for memory
            for i, pair in enumerate(self.fusion_strat_helper.optimized_traces_mem_benchmark_only):
                # Unpack the dict
                label = list(pair.keys())[0]
                trace = list(pair.values())[0]

                trace_time, trace_mem, res = benchmark_trace(trace, self.benchmark_iters)
                del res
                self.debug_msg += (
                    f"Trace name = [{label}] - Target: MEM - Mem = {trace_mem / (2**30)} GB - Time = {trace_time} ms\n{trace}\n\n"
                )
                log(
                    f'Benchmark trace (target MEM) "{label}" (time = {trace_time} ms, mem = {trace_mem / (2**30)} GB:\n{trace}', level=LogLevel.INFO
                )
                if trace_mem < memory_result.mem:
                    memory_result.tm = trace_time
                    memory_result.mem = trace_mem
                    memory_result.trace = trace
                    memory_result.label = label
                    memory_result.index = i

            log(
                f'Benchmark end: Best trace time "{time_result.label} (time = {time_result.tm} ms)":\n{time_result.trace}', level=LogLevel.INFO)
            log(
                f'Benchmark end: Best trace mem "{memory_result.label} (mem = {memory_result.mem / (2 ** 30)} GB)":\n{memory_result.trace}', level=LogLevel.INFO)

            # TODO (matteochen): remove this
            # log(f"Strat comparison: {self.trace_type}")
            # c, m, o = benchmark_trace(tm.trace)
            # del o
            # log(f"best time: {c} ms,  {m/(2**30)} GB")
            # c, m, o = benchmark_trace(mem.trace)
            # del o
            # log(f"best mem: {c} ms,  {m/(2**30)} GB")

            # Here we have to recover the traces without the pass through remat in order to be compliant
            # with thunder flow as we might have request for no remat
            match self.optimization_algorithm:
                case OptimizationAlgorithm.BEST_FUSER:
                    # Unpack dict
                    trc = list(self.fusion_strat_helper.optimized_traces_time[time_result.index].values())[0]
                    self.bw_trace_candidates.attach_best_time_candidate(trc)

                    # Unpack dict
                    trc = list(self.fusion_strat_helper.optimized_traces_mem[memory_result.index].values())[0]
                    self.bw_trace_candidates.attach_best_mem_candidate(trc)

            log(self.bw_trace_candidates.__repr__(), level=LogLevel.DEBUG)

            # Now, finally build the pair fw and bw traces for the requested strat
            # The current fw trace is set by the caller and we take it as is. All current bw traces optimizations are made with the fw trace set by the caller
            forward_time, forward_memory, _ = benchmark_trace(
                self.active_fw_trace, self.benchmark_iters)
            match self.optimizer_type:
                case OptimizerType.RUNTIME:
                    # Used the computed benchmark from above
                    if time_result.tm < memory_result.tm:
                        log(
                            f"out candidate times: (fw){forward_time} ms, (bw){time_result.tm} ms", level=LogLevel.INFO)
                        self.out.append(
                            FinalOutputCandidates(
                                fw=self.active_fw_trace,
                                bw=self.bw_trace_candidates.best_time,
                                cost=forward_time + time_result.tm,
                            )
                        )
                    else:
                        log(
                            f"out candidate times: (fw){forward_time} ms, (bw){memory_result.tm} ms", level=LogLevel.INFO)
                        self.out.append(
                            FinalOutputCandidates(
                                fw=self.active_fw_trace,
                                bw=self.bw_trace_candidates.best_mem,
                                cost=forward_time + memory_result.tm,
                            )
                        )
                case OptimizerType.MEMORY:
                    # Used the computed benchmark from above
                    if time_result.mem < memory_result.mem:
                        log(
                            f"out candidate mem: (fw){forward_memory / (2**30)} GB, (bw){time_result.mem / (2**30)} GB", level=LogLevel.INFO)
                        self.out.append(
                            FinalOutputCandidates(
                                fw=self.active_fw_trace,
                                bw=self.bw_trace_candidates.best_time,
                                cost=forward_memory + time_result.mem,
                            )
                        )
                    else:
                        log(
                            f"out candidate mem: (fw){forward_memory / (2**30)} GB, (bw){memory_result.mem / (2**30)} GB", level=LogLevel.INFO)
                        self.out.append(
                            FinalOutputCandidates(
                                fw=self.active_fw_trace,
                                bw=self.bw_trace_candidates.best_mem,
                                cost=forward_memory + memory_result.mem,
                            )
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


def return_not_used_vars(trace_in: TraceCtx) -> list[TensorProxy]:
    def is_in_sequence(seq: Sequence[Any], t: TensorProxy):
        for e in seq:
            if isinstance(e, TensorProxy) and e.name == t.name:
                return True
        return False

    # Check if this naming is always valid
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


# TODO (matteochen): move into utils module
def benchmark_trace(
        trace: TraceCtx, iters: int = 1, show_func=False, apply_del_last_used=True, snapshot=False, snapshot_name="", nvsight: bool = False, nvsight_fn_name: str = ""
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
            # Warm up cycles
            for _ in range(warm_up_iters):
                fn(*args)
            # Benchmark
            torch.cuda.cudart().cudaProfilerStart()
            for i in range(iters):
                torch.cuda.nvtx.range_push(f'{nvsight_fn_name}-iter{i}')
                fn(*args)
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

            return float('inf'), float('inf'), None
        except Exception as e:
            import inspect
            trc = inspect.getsource(fn)
            print(f"#NVSIGHT FN EXECUTION FAILED:\n{trc}")
            raise e

    def compute_time_cost_ms(fn: Callable, iters: int, *args) -> tuple[float, float, Any]:
        try:
            warm_up_iters = 50
            out = None
            torch.cuda.empty_cache()

            start_events = [torch.cuda.Event(
                enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True)
                          for _ in range(iters)]

            # Warm up cycles
            for _ in range(warm_up_iters):
                fn(*args)
            # Snapshot request
            if snapshot:
                torch.cuda.memory._record_memory_history()
                fn(*args)
                torch.cuda.memory._dump_snapshot(
                    snapshot_name + "_benchmark.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
            # Benchmark
            stream = torch.cuda.current_stream()
            max_allocated_bytes = 0
            torch.cuda.synchronize()
            for i in range(iters):
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                torch.cuda.empty_cache()
                torch.cuda._sleep(1_000_000)
                start_events[i].record(stream)
                fn(*args)
                end_events[i].record(stream)
                max_allocated_bytes = max(
                    max_allocated_bytes, torch.cuda.max_memory_allocated(
                        torch.cuda.current_device())
                )

            torch.cuda.synchronize()
            times = [s.elapsed_time(e)
                     for s, e in zip(start_events, end_events)]
            tot_time = sum(times) / iters
            return tot_time, max_allocated_bytes, out
        except Exception as e:
            import inspect

            trc = inspect.getsource(fn)
            print(f"#FN EXECUTION FAILED:\n{trc}")
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
                else:
                    raise AssertionError(
                        f"Input arg type not recognized: {type(e)}")
        return tuple(res)

    def transform_tensor(arg: TensorProxy) -> torch.Tensor:
        from thunder.core.dtypes import is_float_dtype, is_signedinteger_dtype

        # TODO (matteochen): Missing parallel and fsdp handling...
        # TODO (matteochen): Missing support for meta types ...
        dtype = arg.dtype
        shape = arg.shape
        device = arg.device
        requires_grad = arg.requires_grad
        if dtype is not None and is_float_dtype(dtype):
            torch_dtype = thunder_to_torch_float_dtype(dtype, dtype.bytes)
            tensor: torch.Tensor = torch.randn(
                *shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad
            )
        elif dtype is not None and is_signedinteger_dtype(dtype):
            torch_dtype = thunder_to_torch_int_dtype(dtype.bytes)
            tensor: torch.Tensor = torch.randint(
                0, 10, shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad
            )
        elif dtype is not None and is_boolean_dtype(dtype):
            # TODO (matteochen): maybe random?
            tensor: torch.Tensor = torch.zeros(
                *shape, dtype=torch.bool, device=device.device_str(), requires_grad=requires_grad
            )
        else:
            raise AssertionError(f"dtype {dtype} not supported yet")

        return tensor

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
                    input_args.append(
                        False if arg.value is None else arg.value)
                else:
                    input_args.append(0 if arg.value is None else arg.value)
            elif isinstance(arg, FloatProxy):
                input_args.append(0.0 if arg.value is None else arg.value)
            else:
                raise AssertionError(
                    f"Input arg type not recognized: {type(arg)}")
    else:
        raise AssertionError("Unexpexcted args type")

    if apply_del_last_used:
        trace = del_last_used(trace)

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

    trace_tok = set_tracectx(trace)

    # Obtain the python executable string
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
            t, m, answer = compute_time_cost_ms(executable, iters, *input_args)
    except Exception as e:
        # https://github.com/Lightning-AI/lightning-thunder/issues/664
        print(f"Exception:\n{e}")
        if "call_method UserDefinedObjectVariable(set) __contains__ [UserDefinedObjectVariable()] {}" in str(e) and not nvsight:
            print(
                "Executing with torch compile no full graph (this might still fail), see: https://github.com/Lightning-AI/lightning-thunder/issues/664"
            )
            torch_compiled = torch.compile(executable, fullgraph=False)
            try:
                t, m, answer = compute_time_cost_ms(
                    torch_compiled, iters, *input_args)
            except Exception as e:
                print(f"Compiled trace execution still failed:\n{e}")
        else:
            print(f"Unknown exception occured:\n{e}")
    finally:
        reset_tracectx(trace_tok)

    return t, m, answer

