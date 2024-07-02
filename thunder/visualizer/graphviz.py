import graphviz
from thunder.executors.data_dependent_partition import Node, Graph

def to_graphviz_dag(g: Graph) -> graphviz.Digraph:
    dot = graphviz.Digraph()
    visit_stack = list(g.roots)
    # Add root nodes
    r: Node
    for r in g.roots:
        dot.node(f'{r.ID}({r.group_bsyms[0].sym.name})')

    visited = set()
    cur: Node
    while visit_stack:
        cur = visit_stack.pop(0)
        if cur in visited:
            continue

        cur_node_str = f'{cur.ID}({cur.group_bsyms[0].sym.name})'
        dot.node(cur_node_str)

        # Connect with parent
        for p in cur.parents:
            id = p.ID
            op = p.group_bsyms[0].sym.name
            parent_str = f'{id}({op})'
            dot.edge(parent_str, cur_node_str)

        visited.add(cur)
        visit_stack.extend(cur.children)
    return dot


def create_graphviz_pdf(g: Graph, name='graph'):
    dot = to_graphviz_dag(g)
    dot.render(name, view=False)

