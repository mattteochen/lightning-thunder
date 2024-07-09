from typing import Any, Hashable
import torch
import thunder
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.utils import check, safe_map_flat
from thunder.core.proxies import Proxy, TensorProxy, variableify, Variable
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.executors.data_dependent_partition import Graph, Node
from thunder.core.trace import set_tracectx, reset_tracectx, get_tracectx, TraceCtx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor, get_always_executors
import thunder.core.transforms as transforms
from thunder.visualizer.visualizer_helper import Visualizer
from collections.abc import Callable, Sequence
from enum import Enum
from itertools import chain
import time
# import pprint

class OptimizerNode():
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmark: float):
        self.candidate_executors[ex] = benchmark

class BackendOptimizer():
    def __init__(self, trace: TraceCtx, executors: Sequence[Executor], produce_log=True, log_file_name='autotune_traces_computation_time.log', visualizer: Visualizer | None = None) -> None:
        self.trace: TraceCtx = trace
        self.optimal_trace: TraceCtx = trace
        self.computation_graph: Graph = Graph(trace)
        self.executors: Sequence[Executor] = executors
        self.empty_executor_hashable_placeholder: str = 'empty'
        self.placement_options: list[list[Executor]] = []
        self.optimzide_traces: list[TraceCtx] = []
        self.always_executors: tuple[Executor, ...] = get_always_executors()
        self.produce_log: bool = produce_log
        self.log_file_name: str = log_file_name
        self.log_str: str = ""
        self.visualizer: Visualizer | None = visualizer

        print('INIT TRACE')
        import pprint
        pprint.pprint(self.trace)

    # TODO (matteochen): fix this
    def __repr__(self) -> str:
        return ''

    def write(self, file_name):
        with open(file_name, 'w') as file:
            s = self.__repr__()
            file.write(s)
            file.close()

    def compute_time_cost(self, fn: Callable, iters: int, *args) -> tuple[float, Any]:
        total_time = 0
        out = None
        for _ in range(iters):
            time_s = time.time_ns()
            out = fn(*args)
            time_e = time.time_ns()
            total_time += (time_e - time_s)

        return total_time / iters, out

    def build_placement_options(self):
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
                    print(f'reached end of search for this tree branch {configuration}')
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
        bound_symbols_name = [s.sym.name for s in bound_symbols]
        max_len = len(bound_symbols)

        all_configurations: list[list[Executor]] = []
        # Is the name reserved?
        empty_executor = Executor(name=self.empty_executor_hashable_placeholder)

        print(f'input trace bound symbols name len {len(bound_symbols_name)}: {bound_symbols_name}')

        if len(bound_symbols) > 0:
            search(SearchNode(bound_symbols[0], 0), [])
            print('end of search')
            self.placement_options = all_configurations
            print('config len', len(all_configurations))
            # for config in all_configurations:
            #     c_str = [str(c.name) for c in config]
            #     c_str = " ".join(c_str)
            #     print(c_str)

    def place_optimizers(self, executor_list: list[Executor]) -> TraceCtx:

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

        # for s, o in zip(self.trace.bound_symbols, executor_list):
        #     print(f'{s} -> {o}')

        extrace = transforms.visitor_transform_paired(self.trace, visit, zip(self.trace.bound_symbols, executor_list))

        # Restores original variables
        bound_symbols: list[BoundSymbol] = []
        for bsym in extrace.bound_symbols:
            nbsym: BoundSymbol = bsym.from_bsym_swap_proxies(swapmap)
            bound_symbols.append(nbsym)

        extrace.bound_symbols = bound_symbols

        # print('============================================ trace before fusion pass')
        # pprint.pprint(extrace)

        # We have to temporary clear the subsymbols of already claimed symbols by not fusion ops, otherwise fusion ops will check recursively subsymbols and clear all the current placements
        cached_subsymbols: dict[str, Sequence[BoundSymbolInterface]] = {}
        unique_fusion_executors = set()
        for ex, bsym in zip(executor_list, extrace.bound_symbols):
            bsym_hash: str = hex(id(bsym))
            cached_subsymbols[bsym_hash] = list(bsym.subsymbols)
            if isinstance(ex, FusionExecutor):
                unique_fusion_executors.add(ex)
            else:
                bsym.subsymbols = ()

        # Perform fusion pass
        for ex in unique_fusion_executors:
            extrace = ex.fusion_pass(extrace)

        # Restore the subsymbols
        for bsym in extrace.bound_symbols:
            hash = hex(id(bsym))
            if hash in cached_subsymbols:
                bsym.subsymbols = cached_subsymbols[hash]

        # print('============================================ trace after fusion pass')
        # pprint.pprint(extrace)

        # Apply always executors
        extrace = _transform_for_operator_executor_execution(extrace, self.always_executors)

        # print('============================================ trace after always executors pass')
        # pprint.pprint(extrace)

        return extrace

    def build_search_space(self):
        import thunder.core.codeutils as cutils

        self.build_placement_options()

        for option in self.placement_options:
            option_str = [str(ex.name) for ex in option]
            option_str = '-'.join(option_str)
            # print(f'============================================ optimizers len {len(option)}: {option_str}')
            trace = self.place_optimizers(option)

            if self.visualizer is not None:
                sig_name = cutils.get_siginfo_name(trace)
                # TODO (matteochen): consider adding more infos for naming
                self.visualizer.set_hidden_trace(f'hidden-{sig_name}-{option_str}', trace)

            self.optimzide_traces.append(trace)

    def get_optimal_trace(self) -> TraceCtx:
        return self.optimal_trace

    def benchmark_trace(self, trace: TraceCtx) -> float:

        input_args = []

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

        # print('========================================= benchmark_trace: input_args')
        # print_input_args(input_args, level=0)

        # TODO (matteochen): measure time
        trace_tok = set_tracectx(trace)

        # Obtain the python executable string
        executable_str = trace.python_callable()
        t, _ = self.compute_time_cost(executable_str, 10, *input_args)

        reset_tracectx(trace_tok)

        # Note, currently the forward pass returns a tuple:
        # (
        #     dict,
        #     ...
        # )
        # We have to access the dict['output'] in order to get the forward computation result

        if self.produce_log:
            self.log_str += f'Time taken: {t / 1000000}ms\n'
            self.log_str += trace.python()
            self.log_str += '\n#############################################################################################################\n'

        # print('========================================= benchmark_trace out')
        # print_trace_execution_output(out)

        return t

    def benchmark_traces(self):
        min_run_time = float('inf')
        optimal_trace: TraceCtx = self.trace # Assign initial value for unbound errors
        for trace in self.optimzide_traces:
            trace_time = self.benchmark_trace(trace)
            if trace_time < min_run_time:
                min_run_time = trace_time
                optimal_trace = trace

        self.optimal_trace = optimal_trace

        with open(self.log_file_name, 'w') as file:
            file.write(self.log_str)
            file.close()
