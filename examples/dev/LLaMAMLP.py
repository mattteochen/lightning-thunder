import torch
import thunder
import time

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
    a = 4096 * 3
    b = 11008 * 3
    model = LLaMAMLP(a, b)
    x = torch.randn(2, 2048, a, requires_grad=True)

    jmodel = thunder.jit(model)

    tot_time = 0
    iters = 12
    for i in range(iters):
        start = time.perf_counter_ns()
        ans = jmodel(x)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        # Skip the model without cache
        if i > 1:
            tot_time += (end - start)
        print(f'tot time = {(end - start) / 1000000} ms')


    # for t in thunder.last_traces(jmodel):
    #     print(t)
    print(thunder.last_traces(jmodel)[-1])
    print(thunder.last_backward_traces(jmodel)[-1])
    print(f'Mean time = {(tot_time/(iters-2))/1000000} ms')

    print('deviation:', (jmodel(x) - model(x)).abs().max().item())
