from litgpt import GPT
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark
from thunder.tests.litgpt_model import Config
import thunder
import torch

class Test:
    def __init__(self, layers: int, autotune_type: str) -> None:
        self.layers = layers
        self.autotune_type = autotune_type

layers = [Test(2, 'runtime')]

model_name = 'Llama-2-7b-hf'

for test in layers:
    print('Layers:', test.layers)
    cfg = Config.from_name(model_name)
    cfg.n_layer = test.layers
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'):
        model = GPT(cfg)
        x = torch.randint(1, model.config.vocab_size, (1, 512))

        jmodel_def = thunder.jit(model)
        jmodel_auto = thunder.jit(model, autotune_type=test.autotune_type, executors = ['nvfuser', 'torchcompile', 'cudnn', 'torch', 'python'])

        print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
        print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())


        callables = [jmodel_def, ]
        labels = ['def', ]
        inputs = [torch.randint(1, model.config.vocab_size, (1, 512)), ]
        torch_fw_bw_benchmark(callables, model, labels, inputs, True)

        print('Results:')
        traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
        labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
        thunder_fw_bw_benchmark(traces, labels, 10)

        callables = [jmodel_def, jmodel_auto]
        labels = ['def', 'auto']
        inputs = [x, x]
        torch_fw_bw_benchmark(callables, labels, inputs, True)

        print('\n\n\n\n\n\n')
        print(f'{thunder.last_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_traces(jmodel_auto)[-1]}')

        print('\n\n')
        print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
