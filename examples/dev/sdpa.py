import torch
import thunder

class Module(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, query, key, value):
        query = query + query
        key = key * key
        a = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return a


with torch.device('cuda'):
    module = Module()
    j_module = thunder.jit(module)

    query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
    key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
    value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

    ans = j_module(query, key, value)

    print(thunder.last_traces(j_module)[-1])



