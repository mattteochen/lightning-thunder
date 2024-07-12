from typing import Any, Hashable
import torch
import thunder
from thunder.clang import sub
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.utils import check, safe_map_flat
from thunder.core.proxies import Proxy, TensorProxy, variableify, Variable
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.executors.data_dependent_partition import Graph, Node
from thunder.core.trace import from_trace, set_tracectx, reset_tracectx, get_tracectx, TraceCtx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_always_executors
import thunder.core.transforms as transforms
from thunder.visualizer.visualizer_helper import Visualizer
from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain
import time
# import concurrent.futures

class OptimizerNode():
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmark: float):
        self.candidate_executors[ex] = benchmark

class BackendOptimizer():
    def log(self, what: str):
        print(f'================================================================================ Autotune: {what}')

    def __init__(self, trace: TraceCtx, executors: Sequence[Executor], produce_log=True, log_file_name='autotune_traces_computation_time.log', visualizer: Visualizer | None = None) -> None:
        self.trace: TraceCtx = trace
        self.incremental_search_out_trace: TraceCtx
        self.optimal_trace: TraceCtx = trace
        self.computation_graph: Graph = Graph(trace)
        self.executors: Sequence[Executor] = executors
        self.fusion_executors: Sequence[FusionExecutor] = [ex for ex in executors if isinstance(ex, FusionExecutor)]
        self.empty_executor_hashable_placeholder: str = 'empty'
        self.placement_options: list[list[Executor]] = []
        self.optimized_traces: list[dict[str, TraceCtx]] = []
        self.always_executors: tuple[Executor, ...] = get_always_executors()
        self.produce_log: bool = produce_log
        self.log_file_name: str = log_file_name
        self.log_str: str = ""
        self.visualizer: Visualizer | None = visualizer
        self.partial_costs: dict[TraceCtx, float] = {}

        self.log(f'New trace to optimize\n{self.trace}')

    class OptimizationStrat(Enum):
        EXAUSTIVE = 1
        GREEDY = 2

    # TODO (matteochen): fix this
    def __repr__(self) -> str:
        return ''

    def write(self, file_name):
        with open(file_name, 'w') as file:
            s = self.__repr__()
            file.write(s)
            file.close()

    # TODO (matteochen): this has a lot in common with the exaustive search, compact them
    def build_placement_options_incremental(self):
        import sys

        old_max_recursion = sys.getrecursionlimit()
        # TODO (matteochen): parametrize this
        sys.setrecursionlimit(20000)


        class SearchNode:
            def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
                self.symbol = symbol
                self.idx = idx

        # Last index inclusive
        def benchmark_partial_trace(trace_in: TraceCtx, last_idx: int, configuration: list[Executor]) -> tuple[float, TraceCtx]:

            # Retrive all output tensors from each subregion
            tensors = []
            for i in range(last_idx+1):
                if not isinstance(trace_in.bound_symbols[i], BoundSymbol):
                    raise AssertionError('Expected BoundSymbol but received BoundSymbolInterface')
                s = trace_in.bound_symbols[i]
                # For each bsym region we expect to output a Tensor
                tensors.append(s.output)
            # print('Tensors inside partial trace')
            # for t in tensors:
            #     print(t)

            forced_return_bsym = trace_in.bound_symbols[-1].from_bsym(args=tensors) # Should not be an Interface type at this point

            t = from_trace(trace_in)
            # Cut the trace to the required depth
            t.bound_symbols = list(trace_in.bound_symbols)[:last_idx+1]

            t.bound_symbols.append(forced_return_bsym)
            configuration.append(Executor(name=self.empty_executor_hashable_placeholder)) # Empty executor for the forced_return

            # self.log(f'Debug\n{len(t.bound_symbols)}\n{len(exs)}')
            # self.log(f'Debug\n{(t)}\n')

            # Place the assigned symbols
            placed_t = self.place_optimizers(t, configuration)

            cost, answer = benchmark_trace(placed_t)
            del answer
            self.log(f'Executing partial trace for incremental benchmark:\n{placed_t}')
            self.log(f'Symbol under test = {t.bound_symbols[-2].sym.name}')
            self.log(f'Assigned executor = {configuration[-2].name}')
            self.log(f'Time = {cost/1000000} ms')
            # TODO (matteochen): log this to file
            self.partial_costs[t] = cost
            return cost, placed_t

        # We assign an internal id to each symbol based on its idx inside the bound_symbols list
        def search(node: SearchNode, configuration: list[Executor]):

            def continue_search(time_inc: float):
                if node.idx+1 < max_len:
                    new_idx: int = node.idx + 1
                    new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
                    search(SearchNode(new_symbol, new_idx), configuration)
                else:
                    all_configurations.append(configuration)

            has_backend = False
            min_cost = float('inf')
            min_cost_ex = None
            ex: Executor
            # TODO (matteochen): do parallel for
            for ex in self.executors:
                if not isinstance(node.symbol, BoundSymbol):
                    raise AssertionError("Receive a symbol which is not a BoundSymbol")
                if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
                    # print(f'{node.idx}-{ex._name} can execute symbol {node.symbol.sym.name}')
                    # safe_update_dict(node.idx, ExecutorType.OPERATOR, ex)
                    has_backend = True

                    configuration.append(ex)
                    cost, extrace = benchmark_partial_trace(self.trace, node.idx, list(configuration))
                    configuration.pop()

                    if cost < min_cost:
                        min_cost = cost
                        min_cost_ex = ex

                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
                    # print(f'{node.idx}-{ex._name} can fuse symbol {node.symbol.sym.name}')
                    # safe_update_dict(node.idx, ExecutorType.FUSER, ex)
                    has_backend = True

                    configuration.append(ex)
                    cost, extrace = benchmark_partial_trace(self.trace, node.idx, list(configuration))
                    configuration.pop()

                    if cost < min_cost:
                        min_cost = cost
                        min_cost_ex = ex

            if not has_backend:
                configuration.append(empty_executor)
                continue_search(0.0)
            else:
                if min_cost_ex is None:
                    raise AssertionError("Unexpected min cost executor or trace: None")
                self.log(f'For id: {node.idx} - {node.symbol.sym.name} -> best backend {min_cost_ex.name}\n')
                # log_min_cost_trace(min_cost_trace)
                configuration.append(min_cost_ex)
                continue_search(min_cost)

        # res: dict[int, dict[ExecutorType, list[Executor]]] = {}
        bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

        if len(bound_symbols) > 0:
            search(SearchNode(bound_symbols[0], 0), [])
            self.placement_options = all_configurations

        sys.setrecursionlimit(old_max_recursion)

    # TODO (matteochen): this has a lot in common with the exaustive search, compact them
    # def build_placement_options_incremental(self):
    #     class SearchNode:
    #         def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
    #             self.symbol = symbol
    #             self.idx = idx

    #     def retrieve_executors_from_trace(trace_in: TraceCtx, last_symbol_idx:int = -1) -> list[Executor]:
    #         executors: list[Executor] = []
    #         if last_symbol_idx == -1:
    #             last_symbol_idx = len(trace_in.bound_symbols)
    #         for i in range(last_symbol_idx):
    #             if not isinstance(trace_in.bound_symbols[i], BoundSymbol):
    #                 raise AssertionError('Expected BoundSymbol but received BoundSymbolInterface')
    #             s = trace_in.bound_symbols[i]
    #             if s.sym.executor is None:
    #                 executors.append(empty_executor)
    #             else:
    #                 executors.append(s.sym.executor)
    #         return executors

    #     # Last index inclusive
    #     def benchmark_partial_trace(trace_in: TraceCtx, last_idx: int, new_ex: Executor) -> tuple[float, TraceCtx, Any]:

    #         exs: list[Executor] = retrieve_executors_from_trace(trace_in, last_idx)
    #         # for i in range(last_idx):
    #         #     if not isinstance(trace_in.bound_symbols[i], BoundSymbol):
    #         #         raise AssertionError('Expected BoundSymbol but received BoundSymbolInterface')
    #         #     s = trace_in.bound_symbols[i]
    #         #     if s.sym.executor is None:
    #         #         exs.append(empty_executor)
    #         #     else:
    #         #         exs.append(s.sym.executor)
    #         exs.append(new_ex)

    #         # Retrive all output tensors from each subregion
    #         tensors = []
    #         for i in range(last_idx+1):
    #             if not isinstance(trace_in.bound_symbols[i], BoundSymbol):
    #                 raise AssertionError('Expected BoundSymbol but received BoundSymbolInterface')
    #             s = trace_in.bound_symbols[i]
    #             # For each bsym region we expect to output a Tensor
    #             tensors.append(s.output)
    #         # print('Tensors inside partial trace')
    #         # for t in tensors:
    #         #     print(t)

    #         forced_return_bsym = trace_in.bound_symbols[-1].from_bsym(args=tensors) # Should not be an Interface type at this point

    #         t = from_trace(trace_in)
    #         # Cut the trace to the required depth
    #         t.bound_symbols = list(trace_in.bound_symbols)[:last_idx+1]

    #         t.bound_symbols.append(forced_return_bsym)
    #         exs.append(Executor(name=self.empty_executor_hashable_placeholder)) # Empty executor for the forced_return

    #         # self.log(f'Debug\n{len(t.bound_symbols)}\n{len(exs)}')
    #         # self.log(f'Debug\n{(t)}\n')

    #         # Place the assigned symbols
    #         placed_t = self.place_optimizers(t, exs)

    #         cost, answer = benchmark_trace(placed_t)
    #         self.log(f'Executing partial trace for incremental benchmark:\n{placed_t}')
    #         self.log(f'Symbol under test = {t.bound_symbols[-2].sym.name}')
    #         self.log(f'Assigned executor = {exs[-2].name}')
    #         self.log(f'Time = {cost/1000000} ms')
    #         self.partial_costs[t] = cost
    #         return cost, placed_t, answer

    #     # We assign an internal id to each symbol based on its idx inside the bound_symbols list
    #     def search(node: SearchNode, time_so_far: float):

    #         def continue_search(time_inc: float):
    #             if node.idx+1 < max_len:
    #                 new_idx: int = node.idx + 1
    #                 new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
    #                 search(SearchNode(new_symbol, new_idx), time_so_far + time_inc)
    #             else:
    #                 all_configurations.append(retrieve_executors_from_trace(self.incremental_search_out_trace))
    #                 self.log(f'Incremental search ended:\n{self.incremental_search_out_trace}\n{all_configurations[0]}')

    #         # def safe_update_dict(idx: int, type: ExecutorType, ex: Executor):
    #         #     if idx not in res:
    #         #         res[idx] = {}
    #         #         res[node.idx][type] = [ex]
    #         #     else:
    #         #         if type not in res[idx]:
    #         #             res[node.idx][type] = [ex]
    #         #         else:
    #         #             res[node.idx][type].append(ex)

    #         def log_min_cost_trace(trace: TraceCtx):
    #             self.log(f'Min cost trace:\n{trace}')
    #             b: BoundSymbol
    #             for b in trace.bound_symbols:
    #                 self.log(f'sym = {b.sym.name} , ex = {b.sym.executor}')

    #         def extend_min_cost_trace(trace_in: TraceCtx, idx_from_to_extend: int):
    #             new_items = list(self.trace.bound_symbols[idx_from_to_extend:])
    #             # Remove the mock return statement
    #             trace_in.bound_symbols.pop()
    #             trace_in.bound_symbols.extend(new_items)

    #         def update_self_trace(trace_in: TraceCtx):
    #             self.incremental_search_out_trace = from_trace(trace_in)
    #             self.incremental_search_out_trace.bound_symbols = list(trace_in.bound_symbols)

    #         has_backend = False
    #         min_cost = float('inf')
    #         min_cost_ex = None
    #         min_cost_trace = from_trace(self.incremental_search_out_trace)
    #         min_cost_trace.bound_symbols  = list(self.incremental_search_out_trace.bound_symbols)
    #         # self.log(f'New iter, node idx = {node.idx}')
    #         log_min_cost_trace(min_cost_trace)

    #         trace_iter = from_trace(self.incremental_search_out_trace)
    #         trace_iter.bound_symbols = list(self.incremental_search_out_trace.bound_symbols)

    #         # Seach for last placed executor index in min_cost_trace
    #         idx = 0
    #         while idx < len(min_cost_trace.bound_symbols) and not self.bsym_assigned(min_cost_trace.bound_symbols[idx]):
    #             idx += 1
    #         while idx < len(min_cost_trace.bound_symbols) and self.bsym_assigned(min_cost_trace.bound_symbols[idx]):
    #             idx += 1
    #         # With Fusion operators, our trace will be collapsed. If the min_cost_trace is assigned to a trace that comes out from a fusion pass
    #         # the length of the partial trace (local optimal) to be injected inside benchmark_partial_trace is < node.idx
    #         idx = min(idx, node.idx)

    #         ex: Executor
    #         for ex in self.executors:
    #             if not isinstance(node.symbol, BoundSymbol):
    #                 raise AssertionError("Receive a symbol which is not a BoundSymbol")
    #             if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
    #                 # print(f'{node.idx}-{ex._name} can execute symbol {node.symbol.sym.name}')
    #                 # safe_update_dict(node.idx, ExecutorType.OPERATOR, ex)
    #                 has_backend = True

    #                 cost, extrace, tensor_out = benchmark_partial_trace(self.incremental_search_out_trace, idx, ex)

    #                 if cost < min_cost:
    #                     min_cost = cost
    #                     min_cost_ex = ex
    #                     min_cost_trace = extrace

    #             if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
    #                 # print(f'{node.idx}-{ex._name} can fuse symbol {node.symbol.sym.name}')
    #                 # safe_update_dict(node.idx, ExecutorType.FUSER, ex)
    #                 has_backend = True

    #                 cost, extrace, tensor_out = benchmark_partial_trace(self.incremental_search_out_trace, idx, ex)

    #                 if cost < min_cost:
    #                     min_cost = cost
    #                     min_cost_ex = ex
    #                     min_cost_trace = extrace

    #         if not has_backend:
    #             continue_search(0.0)
    #             # configuration.pop(-1)
    #         else:
    #             if min_cost_ex is None or min_cost_trace is None:
    #                 raise AssertionError("Unexpected min cost executor or trace: None")
    #             self.log(f'For id: {node.idx} - {node.symbol.sym.name} -> best backend {min_cost_ex.name}')
    #             if node.idx + 1 < max_len:
    #                 extend_min_cost_trace(min_cost_trace, node.idx+1)
    #             # log_min_cost_trace(min_cost_trace)
    #             update_self_trace(min_cost_trace)
    #             continue_search(min_cost)

    #     # Assign search initial trace
    #     self.incremental_search_out_trace = from_trace(self.trace)
    #     self.incremental_search_out_trace.bound_symbols = list(self.trace.bound_symbols)

    #     # res: dict[int, dict[ExecutorType, list[Executor]]] = {}
    #     bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
    #     max_len = len(bound_symbols)

    #     all_configurations: list[list[Executor]] = []
    #     # Is the name reserved?
    #     empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

    #     if len(bound_symbols) > 0:
    #         search(SearchNode(bound_symbols[0], 0), 0.0)
    #         self.placement_options = all_configurations

    # This expects a trace after the placement call.
    # Fusion operators as nvFuser can be slower on the single trace region but can be faster by combining more of them,
    # try to fuse then and compare
    def try_to_fuse_after_executors_placement(self, trace_in: TraceCtx) -> TraceCtx:

        def count_fusion_regions(trace_in: TraceCtx) -> int:
            count = 0
            for bsym in trace_in.bound_symbols:
                if isinstance(bsym.sym.executor, FusionExecutor):
                              count += 1
            # ex.fuseion_pass regions are zero indexed
            return max(0, count)

        best_trace: TraceCtx = trace_in
        best_time, answer = benchmark_trace(best_trace)
        del answer
        trace_in_time = best_time

        # for bsym in trace_in.bound_symbols:
        #     print(f'subsymbols: {bsym.subsymbols}')

        fusion_regions = count_fusion_regions(trace_in)
        self.log(f'Try to fuse. Fusion regions already present: {fusion_regions}')

        for ex in self.fusion_executors:
            self.log(f'Try to fuse executor {ex.name} with trace:\n{trace_in}')
            extrace = ex.fusion_pass(trace_in, fusion_regions)
            self.log(f'Fused trace:\n{extrace}')
            extrace_time, answer = benchmark_trace(extrace)
            del answer
            self.log(f'Fused trace time:{extrace_time/1000000} ms')

            if extrace_time < best_time:
                best_time = extrace_time
                best_trace = extrace

        self.log(f'Trace in (time = {trace_in_time / 1000000} ms):\n{trace_in}')
        self.log(f'Best fused trace (time = {best_time / 1000000} ms):\n{best_trace}')

        return best_trace

    def build_placement_options_exaustive(self):
        class ExecutorType(Enum):
            OPERATOR = 1
            FUSER = 1

        class SearchNode:
            def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
                self.symbol = symbol
                self.idx = idx

        # We assign an internal id to each symbol based on its idx inside the bound_symbols list
        def search(node: SearchNode, configuration):
            def continue_search():
                if node.idx+1 < max_len:
                    new_idx: int = node.idx + 1
                    new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
                    search(SearchNode(new_symbol, new_idx), configuration)
                else:
                    # print(f'reached end of search for this tree branch {configuration}')
                    all_configurations.append(list(configuration))

            def safe_update_dict(idx: int, type: ExecutorType, ex: Executor):
                if idx not in res:
                    res[idx] = {}
                    res[node.idx][type] = [ex]
                else:
                    if type not in res[idx]:
                        res[node.idx][type] = [ex]
                    else:
                        res[node.idx][type].append(ex)

            ex: Executor
            has_backend = False
            for ex in self.executors:
                if not isinstance(node.symbol, BoundSymbol):
                    raise AssertionError("Receive a symbol which is not a BoundSymbol")
                if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
                    # print(f'{node.idx}-{ex._name} can execute symbol {node.symbol.sym.name}')
                    safe_update_dict(node.idx, ExecutorType.OPERATOR, ex)
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)
                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
                    # print(f'{node.idx}-{ex._name} can fuse symbol {node.symbol.sym.name}')
                    safe_update_dict(node.idx, ExecutorType.FUSER, ex)
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)

            if not has_backend:
                configuration.append(empty_executor)
                continue_search()
                configuration.pop(-1)

        res: dict[int, dict[ExecutorType, list[Executor]]] = {}
        bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

        if len(bound_symbols) > 0:
            search(SearchNode(bound_symbols[0], 0), [])
            self.placement_options = all_configurations

    # def build_placement_options_parallel(self):
    #     class ExecutorType(Enum):
    #         OPERATOR = 1
    #         FUSER = 1

    #     class SearchNode:
    #         def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
    #             self.symbol = symbol
    #             self.idx = idx

    #     # We assign an internal id to each symbol based on its idx inside the bound_symbols list
    #     def search(node: SearchNode, configuration, all_configurations, level = 0):
    #         def update():
    #             # print(f'{node.idx + 1} >= {max_len}, reached end of search for this tree branch (len = {len(configuration)}) {configuration}')
    #             all_configurations.append(list(configuration))

    #         def safe_update_dict(idx: int, type: ExecutorType, ex: Executor):
    #             if idx not in res:
    #                 res[idx] = {}
    #                 res[node.idx][type] = [ex]
    #             else:
    #                 if type not in res[idx]:
    #                     res[node.idx][type] = [ex]
    #                 else:
    #                     res[node.idx][type].append(ex)

    #         futures = []
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=100) as concurrent_executor:

    #             has_backend = False
    #             new_idx: int = node.idx + 1

    #             if new_idx >= max_len:
    #                 # As this is the last symbol, we expect a return statement by default
    #                 configuration.append(empty_executor)
    #                 update()
    #                 return

    #             new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
    #             new_node = SearchNode(new_symbol, new_idx)

    #             ex: Executor
    #             for ex in self.executors:

    #                 if not isinstance(node.symbol, BoundSymbol):
    #                     raise AssertionError("Receive a symbol which is not a BoundSymbol")
    #                 if (isinstance(ex, OperatorExecutor) and ex.can_execute(node.symbol)):
    #                     safe_update_dict(node.idx, ExecutorType.OPERATOR, ex)
    #                     has_backend = True
    #                     configuration.append(ex)
    #                     futures.append(concurrent_executor.submit(search, new_node, list(configuration), all_configurations, level+1))
    #                     configuration.pop(-1)
    #                 if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
    #                     safe_update_dict(node.idx, ExecutorType.FUSER, ex)
    #                     has_backend = True
    #                     configuration.append(ex)
    #                     futures.append(concurrent_executor.submit(search, new_node, list(configuration), all_configurations, level+1))
    #                     configuration.pop(-1)

    #             if not has_backend:
    #                 configuration.append(empty_executor)
    #                 futures.append(concurrent_executor.submit(search, new_node, list(configuration), all_configurations, level+1))
    #                 configuration.pop(-1)

    #             if level == 0:
    #                 concurrent.futures.wait(futures)

    #     res: dict[int, dict[ExecutorType, list[Executor]]] = {}
    #     bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
    #     bound_symbols_name = [s.sym.name for s in bound_symbols]
    #     max_len = len(bound_symbols)

    #     all: list[list[Executor]] = []
    #     # Is the name reserved?
    #     empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

    #     print(f'input trace bound symbols name len {len(bound_symbols_name)}: {bound_symbols_name}')

    #     import time

    #     if len(bound_symbols) > 0:
    #         start = time.time_ns()
    #         search(SearchNode(bound_symbols[0], 0), [], all)
    #         end = time.time_ns()
    #         print(f'End of search, tot time = {(end - start)/1000000} ms. Configurations len = {len(all)}')
    #         self.placement_options = all
    #         # for config in all_configurations:
    #         #     c_str = [str(c.name) for c in config]
    #         #     c_str = " ".join(c_str)
    #         #     print(c_str)

    def place_optimizers(self, in_trace, executor_list: list[Executor]) -> TraceCtx:

        from thunder.executors.passes import _transform_for_operator_executor_execution

        swapmap: dict[Variable, Proxy] = {}

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

            # if self.bsym_assigned(bsym):
            #     return None
            # if bsym.sym.executor is not None:
            #     return None

            # We have mapped this at previous stages
            if ex.name == self.empty_executor_hashable_placeholder:
                return None
            # The call above represent:
            # if bsym.sym.executor is not None:
            #     return None

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

        # self.log(f'Place optimizer, before fusion pass trace:\n{extrace}')

        # proxy_names_to_ignore = set()
        unique_fusion_executors = set()
        cached_subsymbols: dict[str, Sequence[BoundSymbol]] = {}

        if len(executor_list) != len(extrace.bound_symbols):
            raise AssertionError("Invalid executor - bound_symbols lenght")

        for ex, bsym in zip(executor_list, extrace.bound_symbols):
            if isinstance(ex, FusionExecutor):
                unique_fusion_executors.add(ex)
            elif isinstance(ex, OperatorExecutor):
                if isinstance(bsym.output, TensorProxy):
                    t_proxy_name: str = bsym.output.name
                    cached_subsymbols[t_proxy_name] = list(bsym.subsymbols)
                    # This will leave out these symbols from the fusion pass
                    bsym.subsymbols = []

                    # proxy_names_to_ignore.add(t_proxy.name)

        # self.log(f'To ignore:\n{proxy_names_to_ignore}')

        self.log(f'Before fusion pass trace\n{extrace}')

        # Perform fusion pass
        # TODO (matteochen): filter for the current fusion operator as we wanna find the most efficient one
        for ex in unique_fusion_executors:
            extrace = ex.fusion_pass(extrace)

        self.log(f'After fusion pass trace\n{extrace}')

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

        # Apply always executors
        extrace = _transform_for_operator_executor_execution(extrace, self.always_executors)

        # self.log(f'Place optimizer, after always executors:\n{extrace}')

        return extrace


    def clear_bad_inputs(self, trace_in: TraceCtx):

        def args_eq(a, b) -> bool:
            if len(a) != len(b):
                return False
            for obj_a, obj_b in zip(a, b):
                if type(obj_a) == type(obj_b) and isinstance(obj_a, TensorProxy):
                    if obj_a.name != obj_b.name:
                        return False
            return True

        def clear(bsym: BoundSymbol, input):
            size = len(bsym.subsymbols)
            if size > 0:
                for subsym in bsym.subsymbols:
                    if not args_eq(subsym.args, input):
                        print(f'Sub = {subsym.sym.name} Args = {subsym.args}')
                        print(f'Got subsymbol {subsym.sym.name} with different inputs from {bsym.sym.name}')
                        subsym.args = tuple(list(input))
                        clear(subsym, input)

        # Solve the issue of nvfuser mismatrch input args
        for bsym in trace_in.bound_symbols:
            if bsym.sym.executor is not None:
                print(f'Calling clear for {bsym.sym.name} Args({type(bsym.args)}) = {bsym.args}\n')
                clear(bsym, bsym.args)

    # TODO (matteochen): add config for exaustive search or incremental one
    def optimize(self, strat: OptimizationStrat = OptimizationStrat.GREEDY):
        import thunder.core.codeutils as cutils
        from thunder.executors.passes import transform_for_execution

        def greedy():
            # 1. This builds one option by default
            self.build_placement_options_incremental()

            if len(self.placement_options) != 1:
                raise AssertionError("Unexpected placement options size")

            option = self.placement_options[0]
            # self.log(f'sym len: {len(self.trace.bound_symbols)} options len = {len(option)}')
            # self.log(f'Trace to optimize\n{self.trace}')
            # self.log('Chosen options:')
            # for s, o in zip(self.trace.bound_symbols, option):
            #     print(f'{s.sym.name} -> {o.name}')
            # Place the assigned executors
            self.log(f'Placing optimizers for greedy trace:\n{self.trace}')
            for s, o in zip(self.trace.bound_symbols, option):
                print(f'{s.sym.name} -> {o.name}')
            trace_greedy = self.place_optimizers(self.trace, option)
            self.log(f'Greedy trace:\n{trace_greedy}')

            self.clear_bad_inputs(trace_greedy)

            # Append the unique trace
            self.optimized_traces.append({'greedy': trace_greedy})

            # 2. Try to fuse additional regions from the greedy result
            # Attention, if all the fused traces perform worse that the greedy one, the greedy one is returned
            # TODO (matteochen): ignore a duplicated trace
            trace_greedy_fused = self.try_to_fuse_after_executors_placement(trace_greedy)
            self.optimized_traces.append({'fused_greedy': trace_greedy_fused})

            # 3. Try the priority list approach
            trace_priority = transform_for_execution(self.trace, self.executors)
            self.optimized_traces.append({'priority_list': trace_priority})

            # There are no hidden placements hence do not call the visualizer

        def exaustive():
            # This builds one option by default
            self.build_placement_options_exaustive()

            self.log(f'Placement options size: {len(self.placement_options)}')

            for option in self.placement_options:
                option_str = [str(ex.name) for ex in option]
                option_str = '-'.join(option_str)
                # print(f'============================================ optimizers len {len(option)}: {option_str}')
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
        else:
            raise AssertionError('Optimization strat not implemented')

    def get_optimal_trace(self) -> TraceCtx:
        return self.optimal_trace


    def bsym_assigned(self, bsym: BoundSymbol) -> bool:
        return isinstance(bsym.sym.executor, OperatorExecutor) or isinstance(bsym.sym.executor, FusionExecutor)


    def benchmark_traces(self):
        min_run_time = float('inf')
        optimal_trace: TraceCtx = self.trace # Assign initial value for unbound errors
        best_label = ""

        for trace_info in self.optimized_traces:

            label = None
            trace = None
            for k, v in trace_info.items():
                label = k
                trace = v

            trace_time, res = benchmark_trace(trace)
            del res
            self.log(f'Benchmark trace "{label}" (time = {trace_time / 1000000} ms):\n{trace}')
            if trace_time < min_run_time:
                min_run_time = trace_time
                optimal_trace = trace
                best_label = label

        self.log(f'Benchmark end: Best trace "{best_label}":\n{optimal_trace}')

        self.optimal_trace = optimal_trace

        with open(self.log_file_name, 'w') as file:
            file.write(self.log_str)
            file.close()


# This will benpchmark the input trace with the del_last_used call
def benchmark_trace(trace: TraceCtx) -> tuple[float, Any]:
    from thunder.executors.passes import del_last_used

    input_args = []

    def compute_time_cost(fn: Callable, iters: int, *args) -> tuple[float, Any]:
        total_time = 0
        out = None
        for _ in range(iters):
            time_s = time.time_ns()
            out = fn(*args)
            torch.cuda.synchronize()
            time_e = time.time_ns()
            total_time += (time_e - time_s)

        return total_time / iters, out

    def print_input_args(args, level=0, show_content = False):
        for e in args:
            if isinstance(e, tuple) or isinstance(e, list):
                print_input_args(e, level=level+1)
            else:
                print(f'level {level}', type(e))

    def print_trace_execution_output(out: Any, show_content=False):
        if isinstance(out, tuple):
            for e in out:
                print(f'{type(e)}')
        else:
            print(f'{type(out)}')

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
                # print(f'level {level}', type(e))
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
            # print(f'thunder type: {dtype} torch_dtype: {torch_dtype}')
        else:
            # TODO (matteochen): support other types
            raise AssertionError(f"dtype {dtype} not supported yet")

        shape = arg.shape
        device = arg.device
        requires_grad = arg.requires_grad
        # TODO (matteochen): Missing parallel and fsdp handling...
        # TODO (matteochen): Missing support for meta types ...
        tensor: torch.Tensor = torch.randn(*shape, dtype=torch_dtype, device=device.device_str(), requires_grad=requires_grad)
        # print(f'Adding tensor shape: {tensor.shape} dtype: {tensor.dtype} device: {tensor.device} requires_grad: {tensor.requires_grad}')
        return tensor

    # Can we remove this check?
    if isinstance(trace.args, list):
        for arg in trace.args:
            # print(f'current arg {arg}\ntype {type(arg)}')
            if isinstance(arg, tuple):
                # print('Processig tuple')
                input_args.append(transform_input_tuple(arg))
            elif isinstance(arg, TensorProxy):
                # print('Processig TensorProxy')
                e = transform_tensor(arg)
                input_args.append(e)
            else:
                raise AssertionError(f'Input arg type not recognized: {type(arg)}')
    else:
        raise AssertionError('Unexpexcted args type')

    # Always benchmark trace after a deletion last used pass
    trace = del_last_used(trace)

    # TODO (matteochen): measure time
    trace_tok = set_tracectx(trace)

    # Obtain the python executable string
    executable_str = trace.python_callable()
    # TODO (matteochen): make the iters configurable
    t, answer = compute_time_cost(executable_str, 1, *input_args)

    reset_tracectx(trace_tok)

    return t, answer
