import torch
import thunder
from thunder.benchmarks.utils import (
    thunder_fw_bw_benchmark,
    torch_fw_bw_benchmark,
    torch_fw_bw_benchmark_nvsight,
    torch_total_benchmark,
)


class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.Linear(out_features, in_features),
            torch.nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.linear(x)


with torch.device("cuda"):
    m = 1
    in_features = 4096 * m
    out_features = 4096 * m
    model = Module(in_features, out_features)
    x = torch.randn(768, in_features, requires_grad=True)

    jmodel_def = thunder.jit(model, executors=["transformer_engine"], use_cudagraphs=False)
    jmodel_auto = thunder.jit(
        model,
        autotune_type="runtime",
        executors=[
            "nvfuser",
            "transformer_engine",
        ],
        use_cudagraphs=False,
    )

    y = jmodel_def(x)
    y = jmodel_auto(x)

    iters = 100
    fw_traces = [
        thunder.last_traces(jmodel_def)[-1],
        thunder.last_traces(jmodel_auto)[-1],
    ]
    bw_traces = [
        thunder.last_backward_traces(jmodel_def)[-1],
        thunder.last_backward_traces(jmodel_auto)[-1],
    ]
    fw_labels = ["fw_def", "fw_auto"]
    bw_labels = ["bw_def", "bw_auto"]
    print("Results thunder benchmark:")
    thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters)

    callables = [jmodel_def, jmodel_auto]
    labels = ["def", "auto"]
    inputs = [x, x]
    print("\n\nResults torch benchmark:")
    torch_total_benchmark(callables, labels, inputs, iters)
