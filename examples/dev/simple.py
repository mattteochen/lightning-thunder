import torch
import thunder

class Module(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, x: torch.Tensor):
        a = x + x
        return a

with torch.device('cuda'):
    model = Module()
    x = torch.randn(2, 2)

    jmodel = thunder.jit(model)

    ans = jmodel(x)
    print('---------------------------------------------- all traces')
    for t in thunder.last_traces(jmodel):
        print(t)
        print('##############################################')
    print('---------------------------------------------- ans')
    print(ans)

