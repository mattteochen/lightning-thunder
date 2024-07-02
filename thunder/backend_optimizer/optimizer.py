from typing import Hashable
from thunder.executors.data_dependent_partition import Graph, Node
from thunder.core.trace import TraceCtx
from thunder.extend import Executor, FusionExecutor, OperatorExecutor

class OptimizerNode():
    def __init__(self, node: Node):
        self.node: Node = node
        self.candidate_executors: dict[Hashable, float] = {}

    def add_candidate(self, ex: Executor, benchmarck: float):
        self.candidate_executors[ex] = benchmarck

class BackendOptimizer():
    def __init__(self, trace: TraceCtx, executors: list[Executor]) -> None:
        self.trace = trace
        self.computation_graph = Graph(trace)
        self.executors = executors
        self.memo = {}
        self.default_cost = {}
        self.hash_separator = '#'
        self.dummy_cost = 1
        self.optimizer_nodes = []

    def __repr__(self) -> str:
        ret = self.computation_graph.__repr__()
        ret += "\n"
        n: OptimizerNode
        for n in self.optimizer_nodes:
            ret += str(n.node.ID) + ' ####################################'
            ret += "\n"
            ret += n.candidate_executors.__repr__()
            ret += "\n"
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

    def build_search_space(self):
        visited = set()
        def dfs(node: Node):
            visited.add(node.ID)
            childs = node.children
            node_bsym = node.group_bsyms[0]

            optimizer_node: OptimizerNode = OptimizerNode(node)

            print(f'Node id: {node.ID}, symbol: {node_bsym.sym.name}')

            ex: Executor
            for ex in self.executors:
                print(f'analyzing executor {ex._name}')
                if (isinstance(ex, OperatorExecutor) and ex.can_execute(node_bsym)):
                    print(f'{ex._name} can execute symbol {node_bsym.sym.name}')
                    optimizer_node.add_candidate(ex, 1.0)
                if (isinstance(ex, FusionExecutor) and ex.can_fuse(node_bsym)):
                    print(f'{ex._name} can fuse symbol {node_bsym.sym.name}')
                    optimizer_node.add_candidate(ex, 1.0)

            self.optimizer_nodes.append(optimizer_node)

            if childs:
                for c in childs:
                    if c.ID not in visited:
                        dfs(c)

        for root in self.computation_graph.roots:
            dfs(root)
