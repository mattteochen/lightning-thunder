from litgpt import GPT
from thunder.tests.litgpt_model import Config
import thunder
import torch
from thunder.backend_optimizer.optimizer import benchmark_trace

layers = [4, 8, 16, 32]

for l in layers:
    print('Layers:', l)
    cfg = Config.from_name('Llama-2-7b-hf')
    cfg.n_layer = l
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'):
        model = GPT(cfg)
        x = torch.randint(1, model.config.vocab_size, (1, 512))
        jmodel_def = thunder.jit(model)
        jmodel_auto = thunder.jit(model, autotune_type='runtime')
        y = jmodel_def(x)
        yy = jmodel_auto(x)

        jmodel_def = thunder.jit(model)
        jmodel_auto = thunder.jit(model, autotune_type='runtime')

        y = model(x)
        print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())
        print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())

        print('Results ########################################')
        c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='llama-2-7b-hf_def_fw')
        print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='llama-2-7b-hf_auto_fw')
        print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='llama-2-7b-hf_def_bw')
        print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='llama-2-7b-hf_auto_bw')
        print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')
        del o

        print('\n\n\n\n\n\n')
        print(f'{thunder.last_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_traces(jmodel_auto)[-1]}')

        print('\n\n')
        print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
        print('###############################################################################')
        print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')
