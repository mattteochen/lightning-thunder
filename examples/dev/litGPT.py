from litgpt import GPT
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_nvsight, torch_total_benchmark
from thunder.tests.litgpt_model import Config
import thunder
import torch

class Test:
    def __init__(self, layers: int, autotune_type: str, batch_size: int) -> None:
        self.layers = layers
        self.autotune_type = autotune_type
        self.batch_size = batch_size

layers = [Test(4, 'runtime', 1)]

model_name = 'Llama-3-8B'

for test in layers:
    try:
        print('\n\nLayers:', test.layers)
        cfg = Config.from_name(model_name)
        cfg.n_layer = test.layers
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('cuda'):
            model = GPT(cfg)
            x = torch.randint(1, model.config.vocab_size, (test.batch_size, 512))

            jmodel_def = thunder.jit(model)
            jmodel_auto = thunder.jit(
                model,
                autotune_type=test.autotune_type,
                executors=["nvfuser", "torchcompile", "cudnn", "sdpa", "fa3"],
                use_cudagraphs=False,
            )

            print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
            print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())

            iters = 100
            print(f'Results thunder benchmark ({iters} iters):')
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
            thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters)

            print(f'\n\nResults torch fw bw benchmark ({iters} iters):')
            callables = [jmodel_def, jmodel_auto]
            labels = ['def', 'auto']
            inputs = [x, x]
            torch_fw_bw_benchmark(callables, labels, inputs, iters)
            print(f'\n\nResults torch total benchmark ({iters} iters):')
            torch_total_benchmark(callables, labels, inputs, iters)

            torch_fw_bw_benchmark_nvsight(callables, labels, inputs, iters)

            print('\n\n\n\n\n\n')
            print(f'{thunder.last_traces(jmodel_def)[-1]}')
            print('###############################################################################')
            print(f'{thunder.last_traces(jmodel_auto)[-1]}')

            print('\n\n')
            print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
            print('###############################################################################')
            print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
    except Exception as e:
        print(f'Test failed:\n{e}')
