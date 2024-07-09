from thunder.core.trace import TraceCtx
from thunder.core.transform_common import dce
from thunder.executors.data_dependent_partition import Graph
from thunder.visualizer.graphviz import create_graphviz_pdf

class Visualizer():
    def __init__(self, produce_hidden = False, traces_directory='traces/') -> None:
        self.produce_hidden = produce_hidden
        self.traces: dict[str, TraceCtx] = {}
        self.hidden_traces: dict[str, TraceCtx] = {}
        self.traces_directory = traces_directory

    def set_fw_initial_trace(self, trace: TraceCtx) -> None:
        self.traces['fw_initial'] = dce(trace)

    def set_fw_optimized_trace(self, trace: TraceCtx) -> None:
        self.traces['fw_optimized'] = dce(trace)

    def set_fw_final_trace(self, trace: TraceCtx) -> None:
        self.traces['fw_final'] = dce(trace)

    def set_bw_initial_trace(self, trace: TraceCtx) -> None:
        self.traces['bw_initial'] = dce(trace)

    def set_bw_optimized_trace(self, trace: TraceCtx) -> None:
        self.traces['bw_optimized'] = dce(trace)

    def set_bw_final_trace(self, trace: TraceCtx) -> None:
        self.traces['bw_final'] = dce(trace)

    def set_hidden_trace(self, name: str, trace: TraceCtx) -> None:
        self.traces[name] = dce(trace)

    def produce(self):
        for k, v in self.traces.items():
            try:
                g = Graph(v)
                create_graphviz_pdf(g, k, directory=self.traces_directory)
            except Exception as e:
                print(f"Visualizer failed to produce {k}: {e}")

        if self.produce_hidden:
            for k, v in self.hidden_traces.items():
                try:
                    g = Graph(v)
                    create_graphviz_pdf(g, k, directory=self.traces_directory)
                except Exception as e:
                    print(f"Visualizer failed to produce hidden {k}: {e}")
