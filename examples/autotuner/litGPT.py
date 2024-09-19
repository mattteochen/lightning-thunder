"""
This script benchmarks litGPT models in a easier way wrt thunder.benchmarks.benchmark_litgpt.py with a fake training loop with no optimizers.
"""

from litgpt import GPT
from thunder.benchmarks.utils import torch_total_benchmark, torch_timer_total_benchmark
from thunder.tests.litgpt_model import Config
import thunder
import torch
import time
from pprint import pprint

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class LitGPTModelThunderConfig:
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


to_run = [
    LitGPTModelThunderConfig(
        1,
        "runtime",
        1,
        executors=[
            "cudnn",
            "sdpa",
            "fa3",
            "nvfuser",
            "nvmath",
            "torchcompile",
        ],
    ),
]

for test in to_run:
    try:
        cfg = Config.from_name(test.model_name)
        cfg.n_layer = test.layers
        if test.seq_len != -1:
            cfg.block_size = test.seq_len
        torch.set_default_dtype(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        pprint(cfg)
        print("Batch size:", test.batch_size)
        with torch.device("cuda"):
            model = GPT(cfg)
            x = torch.randint(1, model.config.vocab_size, (test.batch_size, cfg.block_size))
            target = torch.ones_like(x)

            eager = model
            torch_compile = torch.compile(model)
            jmodel_def = thunder.jit(model)
            jmodel_auto = thunder.jit(
                model,
                autotune_type=test.autotune_type,
                executors=test.executors,
                autotune_optimize_common_blocks=test.optimize_transformer_blocks,
                autotune_optimize_common_blocks_min_size=test.optimize_transformer_min_block_size,
            )
            print("deviation def:", (jmodel_def(x) - model(x)).abs().max().item())
            s = time.time_ns()
            print("deviation auto:", (jmodel_auto(x) - model(x)).abs().max().item())
            e = time.time_ns()
            print("Compilation time:", {(e - s) / 1000000000}, "s")

            iters = 100
            callables = [eager, torch_compile, jmodel_def, jmodel_auto]
            labels = ["eager", "torch.compile", "Thunder", "Thunder Autotuner"]
            inputs = [x, x, x, x]
            print(f"\nResults torch total benchmark ({iters} iters):")
            torch_total_benchmark(callables, labels, inputs, iters, torch.nn.functional.cross_entropy)
            print(f"\nResults torch timer benchmark ({iters} iters):")
            torch_timer_total_benchmark(callables, labels, inputs, test.model_name, torch.nn.functional.cross_entropy)
    except Exception as e:
        print(f"Benchmark failed:\n{e}")
        import traceback

        traceback.print_exc()
