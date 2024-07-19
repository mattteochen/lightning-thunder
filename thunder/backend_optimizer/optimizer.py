from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain
from thunder.core.prims import PrimIDs
from thunder.core.utils import check, safe_map_flat
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.proxies import CollectionProxy, Proxy, TensorProxy, variableify, Variable
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import from_trace, set_tracectx, reset_tracectx, get_tracectx, TraceCtx
from thunder.executors.data_dependent_partition import Graph, Node
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_all_executors, get_always_executors, resolve_executors
from thunder.visualizer.visualizer_helper import Visualizer
from typing import Any, Hashable
import thunder
import thunder.core.transforms as transforms
import torch
# import concurrent.futures

class OptimizerType(Enum):
    MEMORY = 1
    RUNTIME = 2

class OptimizerNode():
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmark: float):
        self.candidate_executors[ex] = benchmark

class BackendOptimizer():
    def log(self, what: str):
        print(f'================================================================================ Autotune: {what}')

    def __init__(self, trace: TraceCtx, priority_executors: Sequence[Executor], produce_log=True, log_file_name='autotune_debug.log', visualizer: Visualizer | None = None, optimizer_type: OptimizerType = OptimizerType.RUNTIME) -> None:
        from thunder.core.transform_common import dce
        # Add more supported ones
        self.trace: TraceCtx = dce(trace)
        self.always_executors: tuple[Executor, ...] = get_always_executors()
        self.computation_graph: Graph = Graph(trace)
        self.debug_msg: str = ""
        self.empty_executor_hashable_placeholder: str = 'empty'
        self.executors: Sequence[Executor] = resolve_executors(['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])
        self.fusion_executors: Sequence[FusionExecutor] = [ex for ex in self.executors if isinstance(ex, FusionExecutor)]
        self.incremental_search_out_trace: TraceCtx
        self.log_file_name: str = log_file_name
        self.optimal_trace_mem: TraceCtx = trace
        self.optimal_trace_time: TraceCtx = trace
        self.optimized_traces_mem: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_mem_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_time: list[dict[str | Hashable, TraceCtx]] = []
        self.optimized_traces_time_benchmark_only: list[dict[str | Hashable, TraceCtx]] = []
        self.partial_costs: dict[TraceCtx, float] = {}
        self.placement_options_mem: list[list[Executor]] = []
        self.placement_options_time: list[list[Executor]] = []
        self.priority_executors: Sequence[Executor] = priority_executors
        self.produce_log: bool = produce_log
        self.strat = None
        self.supported_fusion_executors_by_fusion_strat: set = set(['nvfuser', 'torchcompile'])
        self.visualizer: Visualizer | None = visualizer
        self.optimizer_type: OptimizerType = optimizer_type

        self.log(f'New trace to optimize (strat = {self.optimizer_type}):\n{self.trace}')
        self.log('Executors:')
        for o in self.executors:
            self.log(f'{o.name} -> is operator = {isinstance(o, OperatorExecutor)}, is fusion = {isinstance(o, FusionExecutor)}')

    class OptimizationStrat(Enum):
        EXAUSTIVE = 1
        GREEDY = 2
        BEST_FUSER = 3

    # TODO (matteochen): fix this
    def __repr__(self) -> str:
        return ''

    def write(self, file_name):
        with open(file_name, 'w') as file:
            s = self.__repr__()
            file.write(s)
            file.close()

    class SearchNode:
        def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
            self.symbol = symbol
            self.idx = idx

    class TraceType(Enum):
        COMPUTATIONAL = 0
        FW = 1
        BW = 2

    # TODO (matteochen): this has a lot in common with the exaustive search, compact them
    def build_placement_options_incremental(self, whoami: TraceType = TraceType.COMPUTATIONAL):
        import sys

        old_max_recursion = sys.getrecursionlimit()
        sys.setrecursionlimit(2000)

        # Last index inclusive
        def benchmark_partial_trace(trace_in: TraceCtx, last_idx: int, configuration: list[Executor]) -> tuple[float, TraceCtx]:

            def safe_update_dict(d: dict, key_one, key_two, value):
                if key_one not in d:
                    d[key_one] = {
                        key_two: value
                    }
                else:
                    d[key_one][key_two] = value

            # Retrive all output tensors from each subregion
            tensors = []
            for i in range(last_idx+1):
                if not isinstance(trace_in.bound_symbols[i], BoundSymbol):
                    raise AssertionError('Expected BoundSymbol but received BoundSymbolInterface')
                s = trace_in.bound_symbols[i]
                # For each bsym region we expect to output a Tensor
                tensors.append(s.output)

            forced_return_bsym = trace_in.bound_symbols[-1].from_bsym(args=tensors) # Should not be an Interface type at this point

            t = from_trace(trace_in)
            # Cut the trace to the required depth
            t.bound_symbols = list(trace_in.bound_symbols)[:last_idx+1]

            t.bound_symbols.append(forced_return_bsym)
            configuration.append(Executor(name=self.empty_executor_hashable_placeholder)) # Empty executor for the forced_return

            # Place the assigned symbols
            placed_t = self.place_optimizers(t, configuration)

            cost, mem, answer = benchmark_trace(placed_t, iters=5)
            del answer
            self.log(f'Executing partial trace for incremental benchmark:\n{placed_t}')
            self.log(f'Symbol under test = {t.bound_symbols[-2].sym.name}')
            self.log(f'Assigned executor = {configuration[-2].name}')
            self.log(f'Time = {cost} ms')
            # TODO (matteochen): log this to file
            safe_update_dict(self.partial_costs, whoami, t, cost)
            return cost, placed_t

        # We assign an internal id to each symbol based on its idx inside the bound_symbols list
        def search(node: self.SearchNode, configuration: list[Executor]):

            def continue_search():
                if node.idx+1 < max_len:
                    new_idx: int = node.idx + 1
                    new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
                    search(self.SearchNode(new_symbol, new_idx), configuration)
                else:
                    all_configurations.append(configuration)

            has_backend = False
            min_cost = float('inf')
            min_cost_ex = None
            ex: Executor
            # TODO (matteochen): do parallel for
            for ex in self.executors:
                cost = float('inf')
                if not isinstance(node.symbol, BoundSymbol):
                    raise AssertionError("Receive a symbol which is not a BoundSymbol")
                if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
                    has_backend = True

                    configuration.append(ex)
                    cost, _ = benchmark_partial_trace(self.trace, node.idx, list(configuration))
                    configuration.pop()

                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
                    has_backend = True

                    configuration.append(ex)
                    cost, _ = benchmark_partial_trace(self.trace, node.idx, list(configuration))
                    configuration.pop()

                if cost < min_cost:
                    min_cost = cost
                    min_cost_ex = ex

            if not has_backend:
                configuration.append(empty_executor)
                continue_search()
            else:
                if min_cost_ex is None:
                    raise AssertionError("Unexpected min cost executor or trace: None")
                self.log(f'\nFor id: {node.idx} - {node.symbol.sym.name} -> best backend {min_cost_ex.name}\n')
                configuration.append(min_cost_ex)
                continue_search()

        bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

        if len(bound_symbols) > 0:
            search(self.SearchNode(bound_symbols[0], 0), [])
            self.placement_options = all_configurations

        sys.setrecursionlimit(old_max_recursion)

    # This expects a trace after the placement call.
    # Fusion operators as nvFuser can be slower on the single trace region but can be faster by combining more of them,
    # try to fuse then and compare
    def try_to_fuse_after_executors_placement(self, trace_in: TraceCtx) -> TraceCtx:

        best_trace: TraceCtx = trace_in
        best_time, best_mem, answer = benchmark_trace(best_trace, iters=10)
        del answer
        trace_in_time = best_time

        for ex in self.fusion_executors:
            self.log(f'Try to fuse executor {ex.name} with trace:\n{trace_in}')
            extrace = ex.fusion_pass(trace_in)
            self.log(f'Fused trace:\n{extrace}')
            extrace_time, extrace_mem, answer = benchmark_trace(extrace, iters=10)
            del answer
            self.log(f'Fused trace time:{extrace_time} ms')

            if extrace_time < best_time:
                best_time = extrace_time
                best_trace = extrace

        self.log(f'Trace in (time = {trace_in_time } ms):\n{trace_in}')
        self.log(f'Best fused trace (time = {best_time } ms):\n{best_trace}')

        return best_trace

    def build_placement_options_exaustive(self):

        # We assign an internal id to each symbol based on its idx inside the bound_symbols list
        def search(node: self.SearchNode, configuration):
            def continue_search():
                if node.idx+1 < max_len:
                    new_idx: int = node.idx + 1
                    new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
                    search(self.SearchNode(new_symbol, new_idx), configuration)
                else:
                    all_configurations.append(list(configuration))

            ex: Executor
            has_backend = False
            for ex in self.executors:
                if not isinstance(node.symbol, BoundSymbol):
                    raise AssertionError("Receive a symbol which is not a BoundSymbol")
                if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)
                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)

            if not has_backend:
                configuration.append(empty_executor)
                continue_search()
                configuration.pop(-1)

        bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

        if len(bound_symbols) > 0:
            search(self.SearchNode(bound_symbols[0], 0), [])
            self.placement_options = all_configurations

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
                            raise AssertionError(f'What do you want to do here:\nobj_a:\n{obj_a}\nobj_b:{obj_b}')
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
                raise AssertionError('None trace context')
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

            execution_transform: None | Callable = ex.get_execution_transform(bsym.sym)
            out: Any
            # TODO: What is this?
            if execution_transform is not None:
                out = execution_transform(*bsym.args, **bsym.kwargs)
            elif isinstance(ex, OperatorExecutor):
                # Calls the operator executor's operation
                op: Symbol | None = ex.implmap[bsym.sym.id].symbol
                if op is None:
                    raise AssertionError('op is None')
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

        extrace = transforms.visitor_transform_paired(in_trace, visit, zip(in_trace.bound_symbols, executor_list))

        # Restores original variables
        bound_symbols: list[BoundSymbol] = []
        for bsym in extrace.bound_symbols:
            nbsym: BoundSymbol = bsym.from_bsym_swap_proxies(swapmap)
            bound_symbols.append(nbsym)

        extrace.bound_symbols = bound_symbols

        unique_fusion_executors = set()
        cached_subsymbols: dict[str, Sequence[BoundSymbol]] = {}

        if len(executor_list) != len(extrace.bound_symbols):
            raise AssertionError("len(executor_list) != len(extrace.bound_symbols)")

        for ex, bsym in zip(executor_list, extrace.bound_symbols):
            if isinstance(ex, FusionExecutor):
                unique_fusion_executors.add(ex)
            elif isinstance(ex, OperatorExecutor):
                if isinstance(bsym.output, TensorProxy):
                    t_proxy_name: str = bsym.output.name
                    cached_subsymbols[t_proxy_name] = list(bsym.subsymbols)
                    # This will leave out these symbols from the fusion pass
                    bsym.subsymbols = []

        # Perform fusion pass
        # TODO (matteochen): filter for the current fusion operator as we wanna find the most efficient one
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
        extrace = _transform_for_operator_executor_execution(extrace, self.always_executors)

        return extrace

    # TODO (matteochen): add config for exaustive search or incremental one
    def optimize(self, strat: OptimizationStrat = OptimizationStrat.BEST_FUSER):
        import thunder.core.codeutils as cutils

        self.strat = strat

        from thunder.executors.passes import transform_for_execution
        def best_fuser():
            self.build_placement_options_fusion_regions()

            if len(self.placement_options_time) != len(self.fusion_executors):
                raise AssertionError("Unexpected time placement options size")
            if len(self.placement_options_mem) != len(self.fusion_executors):
                raise AssertionError("Unexpected mem placement options size")

            for placement, ex in zip(self.placement_options_time, self.fusion_executors):
                self.optimized_traces_time.append({ex.name: self.place_optimizers(self.trace, placement)})
            for placement, ex in zip(self.placement_options_mem, self.fusion_executors):
                self.optimized_traces_mem.append({ex.name: self.place_optimizers(self.trace, placement)})

        def greedy():
            # 1. This builds one option by default
            self.build_placement_options_incremental()

            if len(self.placement_options) != 1:
                raise AssertionError("Unexpected placement options size")

            option = self.placement_options[0]
            trace_greedy = self.place_optimizers(self.trace, option)
            # Append the unique trace
            self.optimized_traces.append({'greedy': trace_greedy})

            # 2. Try to fuse additional regions from the greedy result
            # Attention, if all the fused traces perform worse that the greedy one, the greedy one is returned
            # TODO (matteochen): ignore a duplicated trace
            trace_greedy_fused = self.try_to_fuse_after_executors_placement(trace_greedy)
            self.optimized_traces.append({'fused_greedy': trace_greedy_fused})

            # 3. Try the priority list approach
            trace_priority = transform_for_execution(self.trace, self.priority_executors)
            self.optimized_traces.append({'priority_list': trace_priority})

            # There are no hidden placements hence do not call the visualizer

        def exaustive():
            # This builds one option by default
            self.build_placement_options_exaustive()

            self.log(f'Placement options size: {len(self.placement_options)}')

            for option in self.placement_options:
                option_str = [str(ex.name) for ex in option]
                option_str = '-'.join(option_str)
                trace = self.place_optimizers(self.trace, option)

                if self.visualizer is not None:
                    sig_name = cutils.get_siginfo_name(trace)
                    # TODO (matteochen): consider adding more infos for naming
                    self.visualizer.set_hidden_trace(f'hidden-{sig_name}-{option_str}', trace)

                self.optimized_traces.append({option_str: trace})

        if strat == self.OptimizationStrat.GREEDY:
            greedy()
        elif strat == self.OptimizationStrat.EXAUSTIVE:
            exaustive()
        elif strat == self.OptimizationStrat.BEST_FUSER:
            best_fuser()
        else:
            raise AssertionError('Optimization strat not implemented')

    def build_placement_options_fusion_regions(self, increment_factor:int = 1):
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols

        def sequence_hash(s: Sequence) -> str:
            name = ""
            for e in s:
                if isinstance(e, CollectionProxy) or isinstance(e, TensorProxy):
                    name += e.name
                elif e is None:
                    name += "None"
                else:
                    raise AssertionError(f'What? type = {type(e)}')
            return name

        # TODO (matteochen): Benchmark the optimal executor and call this optimal
        def get_default_executor(bsym: BoundSymbol):
            for ex in self.executors:
                if isinstance(ex, FusionExecutor):
                    continue
                if ex.can_execute(bsym):
                    return ex
            return Executor(name=self.empty_executor_hashable_placeholder)

        def get_placed_trace(mapping: dict[str, Executor], bound_symbols_in: Sequence[BoundSymbol]):
            self.log(f'Input mapping len = {len(mapping)}:')
            self.log(f'Input bound_symbols len = {len(bound_symbols_in)}:')
            trc = from_trace(self.trace)
            trc.bound_symbols = list(bound_symbols_in)

            # for b in trc.bound_symbols:
            #     print(b.sym.name)

            # print(f'trc:\n{trc}')

            # def find_original_return_tensors(trace_in: TraceCtx) -> list[Any]:
            #     return_bsym = trace_in.bound_symbols[-1]
            #     if return_bsym.sym.name != 'return':
            #         raise AssertionError(f'Expected return symbol got {return_bsym.sym.name}')

            #     ans = []
            #     if isinstance(return_bsym.args, tuple):
            #         # forward trace
            #         if isinstance(return_bsym.args[0], dict):
            #             ans.append(return_bsym.args[0]['output'])
            #         # backward trace
            #         else:
            #             ans.extend([s for s in return_bsym.args if s is not None])
            #     else:
            #         raise AssertionError('Not supported')

            #     return ans

            # def find_last_out_tensor(trace_in: TraceCtx):
            #     m = 0
            #     t = None
            #     for b in trace_in.bound_symbols:
            #         if b.sym.name == 'return':
            #             continue
            #         if isinstance(b.output, TensorProxy):
            #             if is_possible_out(b.output.name) and int(b.output.name[1:]) > m:
            #                 m = int(b.output.name[1:])
            #                 t = b.output
            #         # else:
            #         #     raise AssertionError(f'Not implemented, type = {type(b.output)}')
            #     if t is None:
            #         raise AssertionError('Max tensor output not found')
            #     print(f'max tensor out name: {t}')
            #     return t

            # def is_tensor_in_bsyms(t: TensorProxy | tuple):
            #     def handle_tuple(tup: tuple):
            #         for e in tup:
            #             if isinstance(e, TensorProxy):
            #                 for b in trc.bound_symbols:
            #                     if b is not None:
            #                         if isinstance(b.output, TensorProxy):
            #                             if b.output.name == e.name:
            #                                 return b.output
            #             else:
            #                 raise AssertionError('Not supported')

            #     if isinstance(t, TensorProxy):
            #         for b in trc.bound_symbols:
            #             if b is not None:
            #                 if isinstance(b.output, TensorProxy):
            #                     if b.output.name == t.name:
            #                         return b.output
            #         return None
            #     else:
            #         handle_tuple(t)


            # tensors = []
            # for b in bound_symbols_in:
            #     if isinstance(b.output, TensorProxy):
            #         tensors.append(b.output)
            # We include always the last tensor as output of the partial trace + all the already
            # available out tensor present in the original trace in order to not be discarded from the dce
            # tensors = [find_last_out_tensor(trc)]
            # original_returns = find_original_return_tensors(self.trace)
            # for t in original_returns:
            #     # TODO (matteochen): improve this
            #     res = is_tensor_in_bsyms(t)
            #     if res is not None:
            #         tensors.append(res)

            # For this partial trace we have to return all not used tensors otherwise the dce will cut them out
            tensors = return_not_used(trc)

            forced_return_bsym = self.trace.bound_symbols[-1].from_bsym(args=tensors)

            executor_configuration = []
            empty_executor = Executor(name=self.empty_executor_hashable_placeholder)
            keys = []
            for bsym in  trc.bound_symbols:
                if bsym.sym.name == 'return':
                    raise AssertionError('return statement should not be here')
                    executor_configuration.append(empty_executor)
                    keys.append('return')
                elif isinstance(bsym.output, Sequence):
                    seq_hash = sequence_hash(bsym.output)
                    executor_configuration.append(mapping.get(seq_hash, empty_executor))
                    keys.append(seq_hash)
                elif isinstance(bsym.output, CollectionProxy) or isinstance(bsym.output, TensorProxy):
                    if bsym.output.name not in mapping:
                        raise AssertionError(f'Expected key {bsym.output.name} in mapping {mapping}')
                    executor_configuration.append(mapping[bsym.output.name])
                    keys.append(bsym.output.name)
                else:
                    raise AssertionError(f"Type not handled: {type(bsym.output)}")

            if trc.bound_symbols[-1].sym.name != 'return':
                trc.bound_symbols.append(forced_return_bsym)
                executor_configuration.append(Executor(name=self.empty_executor_hashable_placeholder))
                keys.append('return')

            if len(trc.bound_symbols) != len(executor_configuration) or len(keys) != len(executor_configuration):
                raise AssertionError(f'len trc.bound_symbols ({len(trc.bound_symbols)}) != len executor_configuration ({len(executor_configuration)}) != len keys ({len(keys)})')

            # self.log(f'Before placement trc:\n{trc}')
            placed_trace = self.place_optimizers(trc, executor_configuration)
            return placed_trace, keys, executor_configuration

        ex: FusionExecutor
        for ex in self.fusion_executors:

            if ex.name not in self.supported_fusion_executors_by_fusion_strat:
                raise AssertionError(f'Fusion operator not supported: {ex.name}')

            self.log(f'Searching best placement for fusion executor = {ex.name}')
            # TODO (matteochen): each executor has a custo def
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

            bound_symbol_groups =fuse_bound_symbols(self.trace, _should_fuse_nvfuser if ex.name == 'nvfuser' else _should_fuse_torchcompile)
            self.log(f'Num of groups = {len(bound_symbol_groups)}')

            for group in bound_symbol_groups:
                for sub in group:
                    self.log(f'{sub.sym.name} -> out: {sub.output}')
                if len(group) > 0:
                    print('\n')

            map_time: dict[str, Executor] = {}
            map_mem: dict[str, Executor] = {}
            increasing_symbols = []
            for group_id, group in enumerate(bound_symbol_groups):
                self.log(f'group start = {group[0].sym.name}')
                self.log(f'group end = {group[-1].sym.name}')

                if group[0].sym.name != 'return':
                    increasing_symbols += group

                # Is not a fusion region, get the default executor
                if len(group) < 2:
                    symbol = group[0]
                    self.log(f'--> Single group: {symbol.sym.name}')
                    name = symbol.sym.name
                    ex_for_this = get_default_executor(symbol)
                    if name == 'return':
                        map_time['return'] = ex_for_this
                        map_mem['return'] = ex_for_this
                        # Add the modified return statement at the end of the for loop
                        break
                    elif isinstance(symbol.output, Sequence):
                        map_time[sequence_hash(symbol.output)] = ex_for_this
                        map_mem[sequence_hash(symbol.output)] = ex_for_this
                    elif isinstance(symbol.output, CollectionProxy) or isinstance(symbol.output, TensorProxy):
                        map_time[symbol.output.name] = ex_for_this
                        map_mem[symbol.output.name] = ex_for_this
                    continue

                # Inside groups we should have alwasy tensors as out
                best_res_time = self.Result()
                best_res_mem = self.Result()
                worst_res_time = self.Result()
                worst_res_mem = self.Result()
                worst_res_mem.measure = 0
                worst_res_time.measure = 0

                best_placement_time = None
                best_keys_time = None
                best_placement_mem = None
                best_keys_mem = None
                # Each iteration of this loop will have map_time = map_mem, hence we use and fill only map_time
                # Best time and best mem will be recorded separatedly though
                for i in range(len(group)):
                    # From top to bottom (this will include the whole region)
                    # -> First iteration is the one with fusion region with single element
                    # -> Last iteration gives the complete fusion region
                    for j in range(0, i+1, increment_factor):
                        map_time[group[j].output.name] = ex
                        map_mem[group[j].output.name] = ex
                    for k in range(i+1, len(group), increment_factor):
                        map_time[group[k].output.name] = get_default_executor(group[k])
                        map_mem[group[k].output.name] = get_default_executor(group[k])

                    # Benchmark this placement
                    trc, keys, placements = get_placed_trace(map_time, increasing_symbols)
                    cost, mem, out = benchmark_trace(trc, iters=1)
                    del out
                    self.log(f'Placed trace (cost = {cost} ms, mem = {mem/(2**30)} GB)\n{trc}')
                    if cost < best_res_time.measure:
                        best_res_time.measure = cost
                        best_res_time.trace = trc
                        best_placement_time = placements
                        best_keys_time = keys
                    if cost > worst_res_time.measure:
                        worst_res_time.measure = cost

                    if mem < best_res_mem.measure:
                        best_res_mem.measure = mem
                        best_res_mem.trace = trc
                        best_placement_mem = placements
                        best_keys_mem = keys
                    if mem > worst_res_mem.measure:
                        worst_res_mem.measure = mem

                    # From bottom to up (this will exclude the full region as being handled in the for cycle above)
                    # -> First iteration is the one with len(fusion_region) - 1
                    # -> Last iteration gives no fusion regions
                    # for j in range(0, i+1, increment_factor):
                    #     map_time[group[j].output.name] = get_default_executor(group[j])
                    # for k in range(i+1, len(group), increment_factor):
                    #     map_time[group[k].output.name] = ex

                    # Benchmark this placement
                    # trc, keys, placements = get_placed_trace(map_time, increasing_symbols)
                    # cost, out = benchmark_trace(trc, iters=2)
                    # del out
                    # self.log(f'Placed trace (cost = {cost } ms)\n{trc}')
                    # if cost < best_time:
                    #     best_time = cost
                    #     best_trc = trc
                    #     best_placement = placements
                    #     best_keys = keys
                if best_placement_time is None or best_keys_time is None:
                    raise AssertionError('Failed to get best placement')
                if best_placement_mem is None or best_keys_mem is None:
                    raise AssertionError('Failed to get best placement')

                self.log(f'For group {group_id} best placement with time cost = {best_res_time.measure} ms (worst time = {worst_res_time.measure} ms):\n{best_res_time.trace}')
                self.log(f'For group {group_id} best placement with mem cost = {best_res_mem.measure / (2**30)} GB (worst mem = {worst_res_mem.measure/(2**30)} GB) is:\n{best_res_mem.trace}')

                # for n, p in zip(best_keys, best_placement):
                #     print(f'{n} -> {p.name}')

                # Update our dict
                for n, p in zip(best_keys_time, best_placement_time):
                    map_time |= {n: p}
                # Update our dict
                for n, p in zip(best_keys_mem, best_placement_mem):
                    map_mem |= {n: p}

            # self.log('End of group search')
            # pprint.pprint(map_time)

            # print('map cmp')
            # for k in map_time.keys():
            #     if k not in map_mem:
            #         pprint.pprint(map_time)
            #         pprint.pprint(map_mem)
            #         raise AssertionError(f"cannot find {k}")
            # pprint.pprint(map_time)
            # pprint.pprint(map_mem)

            # Generate the placement
            executors_time = []
            executors_mem = []
            for bsym in  self.trace.bound_symbols:
                if bsym.sym.name == 'return':
                    if 'return' not in map_time or 'return' not in map_mem:
                        raise AssertionError(f'Expected key return in mapping {map_time} and {map_mem}')
                    executors_time.append(map_time['return'])
                    executors_mem.append(map_mem['return'])
                elif isinstance(bsym.output, Sequence):
                    seq_hash = sequence_hash(bsym.output)
                    if seq_hash not in map_time or seq_hash not in map_mem:
                        raise AssertionError(f'Expected key {seq_hash} in mapping {map_time} and {map_mem}')
                    executors_time.append(map_time[seq_hash])
                    executors_mem.append(map_mem[seq_hash])
                elif isinstance(bsym.output, CollectionProxy) or isinstance(bsym.output, TensorProxy):
                    if bsym.output.name not in map_time or bsym.output.name not in map_mem:
                        raise AssertionError(f'Expected key {bsym.output.name} in mapping {map_time} and {map_mem}')
                    executors_time.append(map_time[bsym.output.name])
                    executors_mem.append(map_mem[bsym.output.name])
                else:
                    raise AssertionError(f"Type not handled: {type(bsym.output)}")

            # Swap return bsym otherwise with no call to remat, we will trace the wrong memory occupation
            test_trc = from_trace(self.trace)
            test_trc.bound_symbols = list(self.trace.bound_symbols)
            test_trc.bound_symbols.pop()
            test_trc.bound_symbols.append(self.trace.bound_symbols[-1].from_bsym(args=return_not_used(test_trc)))
            trc = self.place_optimizers(test_trc, executors_mem)
            c, m, o = benchmark_trace(trc)
            del o
            self.log(f'Debug MEM, mem = {m/(2**30)} GB:\n{trc}')
            self.optimized_traces_mem_benchmark_only.append({ex.name: trc})
            trc = self.place_optimizers(test_trc, executors_time)
            c, m, o = benchmark_trace(trc)
            del o
            self.log(f'Debug TIME, time = {c} ms:\n{trc}')
            self.optimized_traces_time_benchmark_only.append({ex.name: trc})

            # Save executors in order to generate real fw and bw trace with correct output
            self.placement_options_time.append(executors_time)
            self.placement_options_mem.append(executors_mem)

    def get_optimal_trace(self) -> TraceCtx:
        if self.optimizer_type == OptimizerType.RUNTIME:
            return self.optimal_trace_time
        else:
            return self.optimal_trace_mem

    def bsym_assigned(self, bsym: BoundSymbol) -> bool:
        return isinstance(bsym.sym.executor, OperatorExecutor) or isinstance(bsym.sym.executor, FusionExecutor)

    class Result:
        def __init__(self) -> None:
            self.measure: float = float('inf')
            self.trace: TraceCtx | None = None
            self.label: str | Hashable = ""
            self.index = -1

    def benchmark_traces(self):

        tm = self.Result()
        mem = self.Result()

        self.debug_msg += 'Traces benchmarks:\n\n'

        source_mem = None
        source_time = None
        if self.strat == self.OptimizationStrat.BEST_FUSER:
            source_mem = self.optimized_traces_mem_benchmark_only
            source_time = self.optimized_traces_time_benchmark_only
        elif self.strat == self.OptimizationStrat.GREEDY:
            source_mem = self.optimized_traces_mem
            source_time = self.optimized_traces_time
        else:
            raise AssertionError('Not supported')

        for i, trace_info in enumerate(source_time):

            label = None
            trace = None
            for k, v in trace_info.items():
                label = k
                trace = v

            trace_time, _, res = benchmark_trace(trace, iters=10)
            del res
            self.debug_msg += f'Trace name = [{label}] - Time = {trace_time} ms\n{trace}\n\n'
            self.log(f'Benchmark trace "{label}" (time = {trace_time} ms:\n{trace}')
            if trace_time < tm.measure:
                tm.measure = trace_time
                tm.trace = trace
                tm.label = label
                tm.index = i

        for i, trace_info in enumerate(source_mem):

            label = None
            trace = None
            for k, v in trace_info.items():
                label = k
                trace = v

            _, trace_mem, res = benchmark_trace(trace, iters=10)
            del res
            self.debug_msg += f'Trace name = [{label}] - Mem = {trace_mem / (2**30)} GB\n{trace}\n\n'
            self.log(f'Benchmark trace "{label}" (mem = {trace_mem / (2 ** 30)} GB):\n{trace}')
            if trace_mem < mem.measure:
                mem.measure = trace_mem
                mem.trace = trace
                mem.label = label
                mem.index = i

        self.log(f'Benchmark end: Best trace time "{tm.label} (time = {tm.measure} ms)":\n{tm.trace}')
        self.log(f'Benchmark end: Best trace mem "{mem.label} (mem = {mem.measure / (2 ** 30)} GB)":\n{mem.trace}')

        self.log('Strat comparison')
        c, m, o = benchmark_trace(tm.trace)
        del o
        self.log(f'best time: {c} ms,  {m/(2**30)} GB')
        c, m, o = benchmark_trace(mem.trace)
        del o
        self.log(f'best mem: {c} ms,  {m/(2**30)} GB')

        # TODO (matteochen): use time or mem strat
        if self.strat == self.OptimizationStrat.GREEDY:
            self.optimal_trace_time = tm.trace
            self.optimal_trace_mem = mem.trace
        elif self.strat == self.OptimizationStrat.BEST_FUSER:
            d = self.optimized_traces_time[tm.index]
            t = None
            for _, v in d.items():
                t = v
            self.optimal_trace_time = t
            d = self.optimized_traces_mem[mem.index]
            t = None
            for _, v in d.items():
                t = v
            self.optimal_trace_mem = t

        self.log(f'Saved best trace time:\n{self.optimal_trace_time}')
        self.log(f'Saved best trace mem:\n{self.optimal_trace_mem}')

        if self.produce_log:
            with open(self.log_file_name, 'w') as file:
                file.write(self.debug_msg)
                file.close()

def return_not_used(trace_in: TraceCtx) -> list[TensorProxy]:
    def is_in_sequence(seq: Sequence[Any], t:TensorProxy):
        for e in seq:
            if isinstance(e, TensorProxy) and e.name == t.name:
                return True
        return False

    # Check if this naming is always valid
    def is_possible_out(name: str):
        if not name.startswith('t'):
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
            if test.args is not None and (isinstance(test.args, tuple) or isinstance(test.args, list)) and is_in_sequence(test.args, b.output):
                f = True
                break
        if not f:
            ans.append(b.output)
    return ans

# This will benchmark the input trace with the del_last_used call
def benchmark_trace(trace: TraceCtx, iters: int = 1, show_func = False, apply_del_last_used = True, snapshot = False, snapshot_name = "") -> tuple[float, float, Any]:
    from thunder.executors.passes import del_last_used
    import inspect

    input_args = []

    if trace.bound_symbols[-1].sym.id != PrimIDs.RETURN:
        raise AssertionError('Missing return statement')

    def compute_time_cost_ms(fn: Callable, iters: int, *args) -> tuple[float, float, Any]:
        warm_up_iters = 3
        out = None
        torch.cuda.empty_cache()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        max_allocated_bytes = 0
        # Warm up cycles
        for _ in range(warm_up_iters):
            fn(*args)
        # Snapshot request
        if snapshot:
            torch.cuda.memory._record_memory_history()
            fn(*args)
            torch.cuda.memory._dump_snapshot(snapshot_name + "_benchmark.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        # Benchmark
        stream = torch.cuda.current_stream()
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record(stream)
            fn(*args)
            end_events[i].record(stream)
            max_allocated_bytes = max(max_allocated_bytes, torch.cuda.max_memory_allocated(torch.cuda.current_device()))

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(times) / iters
        return tot_time, max_allocated_bytes, out

    def print_input_args(args, level=0, show_content = False):
        for e in args:
            if isinstance(e, tuple) or isinstance(e, list):
                print_input_args(e, level=level+1)
            else:
                print(f'level {level}', type(e))

    # def print_trace_execution_output(out: Any, show_content=False):
    #     if isinstance(out, tuple):
    #         for e in out:
    #             print(f'{type(e)}')
    #     else:
    #         print(f'{type(out)}')

    def thunder_to_torch_float_dtype(byte: int) -> torch.dtype:
        if (byte == 2):
            return torch.float16
        elif (byte == 4):
            return torch.float32
        else:
            return torch.float64

    def transform_input_tuple(t: tuple, level=0) -> tuple:
        res = []
        for e in t:
            if type(e) is tuple:
                res.append(transform_input_tuple(e, level+1))
            else:
                if isinstance(e, TensorProxy):
                    res.append(transform_tensor(e))
                else:
                    # TODO (matteochen): support more data types
                    raise AssertionError(f'Input arg type not recognized: {type(e)}')
        return tuple(res)

    def transform_tensor(arg: TensorProxy) -> torch.Tensor:
        dtype = arg.dtype
        if dtype is not None and type(dtype) is thunder.dtypes.floating:
            torch_dtype = thunder_to_torch_float_dtype(dtype.bytes)
        else:
            # TODO (matteochen): support other types
            raise AssertionError(f"dtype {dtype} not supported yet")

        shape = arg.shape
        device = arg.device
        requires_grad = arg.requires_grad
        # TODO (matteochen): Missing parallel and fsdp handling...
        # TODO (matteochen): Missing support for meta types ...
        tensor: torch.Tensor = torch.randn(*shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad)
        return tensor

    # Can we remove this check?
    if isinstance(trace.args, Sequence):
        for arg in trace.args:
            if isinstance(arg, tuple):
                input_args.append(transform_input_tuple(arg))
            elif isinstance(arg, TensorProxy):
                e = transform_tensor(arg)
                input_args.append(e)
            else:
                raise AssertionError(f'Input arg type not recognized: {type(arg)}')
    else:
        raise AssertionError('Unexpexcted args type')

    # Always benchmark trace after a deletion last used pass as the final trace out will passed under this stage
    if apply_del_last_used:
        trace = del_last_used(trace)

    trace_tok = set_tracectx(trace)

    # Obtain the python executable string
    executable = trace.python_callable()
    if show_func:
        print(inspect.getsource(executable))
    t, m, answer = compute_time_cost_ms(executable, iters, *input_args)

    reset_tracectx(trace_tok)

    return t, m, answer
