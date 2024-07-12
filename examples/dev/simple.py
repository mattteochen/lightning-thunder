import torch
import thunder
import time

class Module(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        a = x + x
        # a_silu = self.silu(a)
        b: torch.Tensor = self.linear(a)
        c = b * b
        # c_silu = self.silu(c)
        d = c + c
        return d

with torch.device('cuda'):
    multiplier = 10
    in_features = 20 * multiplier
    out_features = 30 * multiplier
    model = Module(in_features, out_features)
    x = torch.randn(128, in_features)

    jmodel = thunder.jit(model)

    for _ in range(100):
        start = time.time_ns()
        ans = jmodel(x)
        end = time.time_ns()
        # print('---------------------------------------------- all traces')
        # for t in thunder.last_traces(jmodel):
        #     print(t)
        #     print('##############################################')

        print(f'tot time = {(end - start) / 1000000} ms')

