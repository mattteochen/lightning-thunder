"""
This benchmark script is intended to demonstrate the optimizer working on
the single trace region bext executor (when the forward trace symbol will influence the backward trace).

Set the log level at least to INF0 in `thunder/backend_optimizer/optimizer.py`.
"""
import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_total_benchmark

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.set_default_dtype(dtype)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query, key, value):
        a = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        # Make different inputs as happens in a real model
        b = torch.nn.functional.scaled_dot_product_attention(query + query, key + key, value + value)
        return a + b

with torch.device("cuda"):
    model = Model()

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(
        model, autotune_type="runtime", executors=["nvfuser", "cudnn", "sdpa"]
    )

    q = torch.rand(32, 8, 128, 64 * 1, requires_grad=True)
    k = torch.rand(32, 8, 128, 64 * 1, requires_grad=True)
    v = torch.rand(32, 8, 128, 64 * 1, requires_grad=True)

    jmodel_def(q, k, v)
    jmodel_auto(q, k, v)

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
    print("Thunder benchmark:")
    thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters)

    print("\n\n\n\n\n\n")
    print(f"{thunder.last_traces(jmodel_def)[-1]}")
    print("###############################################################################")
    print(f"{thunder.last_traces(jmodel_auto)[-1]}")

    print("\n\n")
    print(f"{thunder.last_backward_traces(jmodel_def)[-1]}")
    print("###############################################################################")
    print(f"{thunder.last_backward_traces(jmodel_auto)[-1]}")
