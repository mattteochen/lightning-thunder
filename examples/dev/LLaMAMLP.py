import torch
import thunder

class LLaMAMLP(torch.nn.Module):
    def __init__(self, n_embd, intermediate_size) -> None:
        super().__init__()
        self.fc_1 = torch.nn.Linear(n_embd, intermediate_size, bias=False)
        self.fc_2 = torch.nn.Linear(n_embd, intermediate_size, bias=False)
        self.proj = torch.nn.Linear(intermediate_size, n_embd, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

with torch.device('cuda'):
    model = LLaMAMLP(4096, 11008)
    x = torch.randn(2, 2048, 4096, requires_grad=True)

    jmodel = thunder.jit(model)

    ans = jmodel(x)
    print('---------------------------------------------- all traces')
    for t in thunder.last_traces(jmodel):
        print(t)
        print('##############################################')
    print('---------------------------------------------- ans')
    print(ans)
