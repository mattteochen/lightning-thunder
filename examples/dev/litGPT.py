from litgpt import GPT
from thunder.tests.litgpt_model import Config
import thunder
import torch
from thunder.backend_optimizer.optimizer import benchmark_trace

cfg = Config.from_name('Llama-2-7b-hf')
cfg.n_layer = 1 # fewer layers
torch.set_default_dtype(torch.bfloat16)

with torch.device('cuda'):
    model = GPT(cfg)
    x = torch.randint(1, model.config.vocab_size, (1, 512))
    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime')
    y = jmodel_def(x)
    yy = jmodel_auto(x)

    jmodel_def = thunder.jit(model)
    # This model fails under some circumstances after passed the placed traced under the rematelizer
    jmodel_auto = thunder.jit(model, autotune_type='memory')

    y = model(x)
    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())
    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())

    print('########################################')
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_def_fw')
    print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_auto_fw')
    print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_def_bw')
    print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
    del o
    c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='MLP_auto_bw')
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
