"""
This script benchmarks litGPT models in a easier wrt to benchmark_litgpt.py way with a fake training loop with no optimizers in order to focus more on
forward and backward computation time and not others kernel during the loop.
"""
from litgpt import GPT
from thunder.benchmarks.utils import (
    thunder_fw_bw_benchmark,
    torch_fw_bw_benchmark,
    torch_fw_bw_benchmark_nvsight,
    torch_total_benchmark,
)
from thunder.tests.litgpt_model import Config
import thunder
import torch
import time

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class Test:
    def __init__(
        self,
        layers: int,
        autotune_type: str,
        batch_size: int,
        seq_len: int = -1,
        model_name: str = "Llama-3-8B",
        executors=None,
        optimize_transformer_blocks=True,
        optimize_transformer_min_block_size=60,  # for llama3
    ) -> None:
        self.layers = layers
        self.autotune_type = autotune_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model_name = model_name
        self.executors = executors
        self.optimize_transformer_blocks = optimize_transformer_blocks
        self.optimize_transformer_min_block_size = optimize_transformer_min_block_size


layers = [
    Test(
        1,
        "runtime",
        1,
        executors=[
            "cudnn",
            "sdpa",
            # "fa3",
            "nvfuser",
            "torchcompile",
        ],
    ),
    Test(
        4,
        "runtime",
        1,
        executors=[
            "cudnn",
            "sdpa",
            # "fa3",
            "nvfuser",
            "torchcompile",
        ],
    ),
]

for test in layers:
    try:
        cfg = Config.from_name(test.model_name)
        cfg.n_layer = test.layers
        if test.seq_len != -1:
            cfg.block_size = test.seq_len
        torch.set_default_dtype(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        print(cfg)
        with torch.device("cuda"):
            model = GPT(cfg)
            x = torch.randint(1, model.config.vocab_size, (test.batch_size, cfg.block_size))
            print(f"Input size: {x.size()}")

            eager = model
            torch_compile = torch.compile(model)
            jmodel_def = thunder.jit(model)
            jmodel_auto = thunder.jit(
                model,
                autotune_type=test.autotune_type,
                executors=test.executors,
                use_cudagraphs=False,
                autotune_optimize_common_blocks=test.optimize_transformer_blocks,
                autotune_optimize_common_blocks_min_size=test.optimize_transformer_min_block_size,
            )
            print("deviation def:", (jmodel_def(x) - model(x)).abs().max().item())
            s = time.time_ns()
            print("deviation auto:", (jmodel_auto(x) - model(x)).abs().max().item())
            e = time.time_ns()
            print("Compilation time:", {(e - s) / 1000000000}, "s")

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
            print("\n\n####################################################", test.model_name)
            print(f"Results thunder benchmark ({iters} iters):")
            thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters, nvsight=False)
            # thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, 10, nvsight=True)

            print(f"\n\nResults torch fw bw benchmark ({iters} iters):")
            callables = [eager, torch_compile, jmodel_def, jmodel_auto]
            labels = ["eager", "torch.compile", "Thunder", "Thunder Autotuner"]
            inputs = [x, x, x, x]
            torch_fw_bw_benchmark(callables, labels, inputs, iters)
            print(f"\n\nResults torch total benchmark ({iters} iters):")
            torch_total_benchmark(callables, labels, inputs, iters)

            torch_fw_bw_benchmark_nvsight(callables, labels, inputs, iters)

            print("\n\n\n\n\n\n")
            print(f"{thunder.last_traces(jmodel_def)[-1]}")
            print("###############################################################################")
            print(f"{thunder.last_traces(jmodel_auto)[-1]}")

            print("\n\n")
            print(f"{thunder.last_backward_traces(jmodel_def)[-1]}")
            print("###############################################################################")
            print(f"{thunder.last_backward_traces(jmodel_auto)[-1]}")
    except Exception as e:
        print(f"Test failed:\n{e}")
        import traceback

        traceback.print_exc()
