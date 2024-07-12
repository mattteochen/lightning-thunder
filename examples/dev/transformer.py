
import torch
import thunder

with torch.device('cuda'):
    transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out)


    jmodel = thunder.jit(transformer_model)
    out = jmodel(src, tgt)

