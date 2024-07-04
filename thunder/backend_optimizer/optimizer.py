from typing import Any, Hashable
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.utils import check, safe_map_flat
from thunder.core.proxies import Proxy, variableify, Variable
from thunder.core.symbol import BoundSymbol
from thunder.executors.data_dependent_partition import Graph, Node
from thunder.core.trace import set_tracectx, reset_tracectx, from_trace, get_tracectx, TraceProvenance, TraceCtx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor
import thunder.core.transforms as transforms
from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain
import time
import pprint

class OptimizerNode():
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmarck: float):
        self.candidate_executors[ex] = benchmarck

class BackendOptimizer():
    def __init__(self, trace: TraceCtx, executors: Sequence[Executor]) -> None:
        self.trace = trace
        self.computation_graph = Graph(trace)
        self.executors = executors
        self.default_cost = {}
        self.hash_separator = '#'
        self.dummy_cost = 1
        self.optimizer_nodes = []
        self.placement_options: list[list[Executor]] = []

    def __repr__(self) -> str:
        ret = self.computation_graph.__repr__()
        ret += "\n"
        n: OptimizerNode
        for n in self.optimizer_nodes:
            ret += f'NODE: {str(n.node.ID)} - {str(n.node.group_bsyms[0].sym.name)} ####################################\n {n.candidate_executors.__repr__()}\n'
        return ret

    def write(self, file_name):
        with open(file_name, 'w') as file:
            s = self.__repr__()
            file.write(s)
            file.close()

    def subgraph_hash(self, nodes: list[Node]):
        ids = [str(n.ID) for n in nodes]
        return self.hash_separator.join(ids)

    # TODO: to implement
    def compute_default_costs_subgraphs(self, nodes: list[Node]):
        hash = self.subgraph_hash(nodes)
        self.default_cost[hash] = self.dummy_cost

    def backed_placer(self):
        pass

    def build_placement_options(self):
        class ExecutorType(Enum):
            OPERATOR = 1
            FUSER = 1

        class SearchNode:
            def __init__(self, symbol: BoundSymbolInterface, idx: int)-> None:
                self.symbol = symbol
                self.idx = idx
                pass

        # We assign an internal id to each symbol based on its idx inside the bound_symbols list
        def search(node: SearchNode, configuration):
            def continue_search():
                if node.idx+1 < max_len:
                    new_idx: int = node.idx + 1
                    new_symbol: BoundSymbolInterface = bound_symbols[new_idx]
                    search(SearchNode(new_symbol, new_idx), configuration)
                else:
                    all_configurations.append(list(configuration))

            def update_dict(idx: int, type: ExecutorType, ex: Executor):
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
                    update_dict(node.idx, ExecutorType.OPERATOR, ex)
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)
                else:
                    pass
                    # print(f'{node.idx}-{ex._name} can NOT execute symbol {node.symbol.sym.name}')
                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node.symbol)):
                    # print(f'{node.idx}-{ex._name} can fuse symbol {node.symbol.sym.name}')
                    update_dict(node.idx, ExecutorType.FUSER, ex)
                    has_backend = True
                    configuration.append(ex)
                    continue_search()
                    configuration.pop(-1)
                else:
                    pass
                    # print(f'{node.idx}-{ex._name} can NOT fuse symbol {node.symbol.sym.name}')

            if not has_backend:
                configuration.append(empty_executor)
                continue_search()
                configuration.pop(-1)

        res: dict[int, dict[ExecutorType, list[Executor]]] = {}
        bound_symbols: list[BoundSymbolInterface] = self.trace.bound_symbols
        bound_symbols_name = [s.sym.name for s in bound_symbols]
        # bound_symbols_id = [s.sym.id for s in bound_symbols]
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name='empty')

        print(f'input trace bound symbols name: {bound_symbols_name}')
        # print(f'input trace bound symbols id: {bound_symbols_id}')

        if len(bound_symbols) > 0:
            search(SearchNode(bound_symbols[0], 0), [])
            self.placement_options = all_configurations
            print(len(all_configurations))
            print('END OF SEDARCH')
            for config in all_configurations:
                c_str = [str(c.name) for c in config]
                c_str = " ".join(c_str)
                print(c_str)

    def place_optimizers(self, executor_list: list[Executor]) -> TraceCtx:
        start_time_ns = time.time_ns()

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
            trace = get_tracectx()
            trace.scopes[-1].append(bsym)
            for p in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
                trace.names.add(p.name)
            return bsym.output

        def visit_helper(bsym: BoundSymbol, ex: Executor) -> None | bool:
            if bsym.sym.python_impl is not None:
                return None

            # We have mapped this at previous stages
            if ex.name == 'empty':
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
                op = ex.implmap[bsym.sym.id].symbol
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

        def visitor_transform(trace_from: TraceCtx, executors: list[Executor]):
            trc: TraceCtx = from_trace(trace_from)

            try:
                tracectx_tok = set_tracectx(trc)

                for bsym, ex in zip(trace_from.bound_symbols, executors):
                    try:
                        # Creates a temporary scope to support copying the original bsym BEFORE
                        #   the operations performed by visit(), even though this doesn't know whether to
                        #   copy the original bsym until after visit() completes
                        old_scope = trc.scopes
                        scope = []
                        trc.scopes = [scope]

                        # This can be simpler? We currently trigger all the flow for the substitution
                        visit_type = visit(bsym, ex)

                        if visit_type is transforms.VISIT_TYPE.INSERT_AFTER:
                            trc.bound_symbols.append(bsym)

                        if visit_type is not transforms.VISIT_TYPE.NO_OP:
                            trc.bound_symbols.extend(scope)
                        else:
                            trc.bound_symbols.append(bsym)

                        if visit_type is transforms.VISIT_TYPE.INSERT_BEFORE:
                            trc.bound_symbols.append(bsym)

                    finally:
                        # Restores the trc's scope
                        trc.scopes = old_scope

                return trc

            finally:
                reset_tracectx(tracectx_tok)

        extrace = visitor_transform(self.trace, executor_list)

        # Restores original variables
        bound_symbols: list[BoundSymbol] = []
        for bsym in extrace.bound_symbols:
            nbsym: BoundSymbol = bsym.from_bsym_swap_proxies(swapmap)
            bound_symbols.append(nbsym)

        extrace.bound_symbols = bound_symbols

        end_time_ns = time.time_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000
        extrace.set_provenance(
            TraceProvenance(f"Transform for operator executor execution (took {elapsed_time_millis} milliseconds)")
        )

        print('============================================ trace before fusion pass')
        pprint.pprint(extrace)

        # We have to temporary clear the subsymbols of already claimed symbols by not fusion ops, otherwise fusion ops will check recursively subsymbols and clear all the current placements
        cached_subsymbols: list[Sequence[BoundSymbolInterface]] = [list(symbol.subsymbols) for symbol in extrace.bound_symbols]
        subsymbols_idx_to_restore: list[int] = []
        unique_fusion_executors = set()
        for idx, ex in enumerate(executor_list):
            if isinstance(ex, FusionExecutor):
                unique_fusion_executors.add(ex)
            else:
                subsymbols_idx_to_restore.append(idx)
                extrace.bound_symbols[idx].subsymbols = ()

        for ex in unique_fusion_executors:
            extrace = ex.fusion_pass(extrace)

        # Restore the subsymbols
        for idx in subsymbols_idx_to_restore:
            extrace.bound_symbols[idx].subsymbols = cached_subsymbols[idx]

        return extrace

    def build_search_space(self):
        self.build_placement_options()
        for option in self.placement_options:
            trace = self.place_optimizers(option)
            option_str = [str(ex.name) for ex in option]
            option_str = ' '.join(option_str)
            print(f'============================================ config_trace, optimizers: {option_str}')
            pprint.pprint(trace)

        # visited = set()
        # def dfs(node: Node):
        #     visited.add(node.ID)
        #     childs = node.children
        #     node_symbols = [str(s.sym.id) for s in node.group_bsyms]
        #     node_symbols = " ".join(node_symbols)
        #     node_bsym = node.group_bsyms[0]

        #     optimizer_node: OptimizerNode = OptimizerNode(node)

        #     print(f'-> Node id: {node.ID}, symbols: {node_symbols}')

        #     ex: Executor
        #     for ex in self.executors:
        #         print(f'analyzing executor {ex._name}')
        #         if (isinstance(ex, OperatorExecutor) and ex.can_execute(node_bsym)):
        #             print(f'{ex._name} can execute symbol {node_bsym.sym.name}')
        #             optimizer_node.add_candidate(ex, 1.0)
        #         if (isinstance(ex, FusionExecutor) and ex.can_fuse(node_bsym)):
        #             print(f'{ex._name} can fuse symbol {node_bsym.sym.name}')
        #             optimizer_node.add_candidate(ex, 1.0)

        #     self.optimizer_nodes.append(optimizer_node)

        #     if childs:
        #         for c in childs:
        #             if c.ID not in visited:
        #                 dfs(c)

        # for root in self.computation_graph.roots:
        #     dfs(root)
