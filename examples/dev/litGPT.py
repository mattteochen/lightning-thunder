from litgpt import GPT
from thunder.tests.litgpt_model import Config
import thunder
import torch

cfg = Config.from_name('Llama-2-7b-hf')
cfg.n_layer = 16 # fewer layers
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):
    m = GPT(cfg)
    thunder_model = thunder.jit(m)

    inp = torch.randint(1, m.config.vocab_size, (1, 512))

    actual = thunder_model(inp)
    expected = m(inp)
