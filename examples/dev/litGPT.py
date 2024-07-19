from litgpt import GPT
from thunder.tests.litgpt_model import Config
import thunder
import torch

cfg = Config.from_name('Llama-2-7b-hf')
cfg.n_layer = 8 # fewer layers
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):
    m = GPT(cfg)
    x = torch.randint(1, m.config.vocab_size, (1, 512))
    jmodel_def = thunder.jit(m)
    jmodel_auto = thunder.jit(m, autotune_type='runtime')
    y = jmodel_def(x)
    yy = jmodel_auto(x)
