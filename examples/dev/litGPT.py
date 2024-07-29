from litgpt import GPT
from thunder.tests.litgpt_model import Config
import thunder
import torch
from thunder.backend_optimizer.optimizer import benchmark_trace

class Test:
    def __init__(self, layers: int, autotune_type: str) -> None:
        self.layers = layers
        self.autotune_type = autotune_type

layers = [Test(1, 'runtime')]

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
        jmodel_auto = thunder.jit(model, autotune_type=test.autotune_type, executors = ['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])

        print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())
        print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())

        print('Results ########################################')
        c, m, _ = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name=model_name)
        print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
        c, m, _ = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name=model_name)
        print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
        c, m, _ = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name=model_name)
        print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
        c, m, _ = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name=model_name)
        print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')

        print('\n\n\n\n\n\n')
        print(f'{thunder.last_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_traces(jmodel_auto)[-1]}')

        print('\n\n')
        print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
