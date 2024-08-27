import torch
import time

iters = 1000

with torch.device("cuda"):
    tot_time = 0
    for i in range(iters):
        s = time.time_ns()
        a = torch.randn(2, 2048, 4096 // 1, requires_grad=True)
        b = torch.randn(2, 2048, 4096 // 1, requires_grad=True)
        c = a + b + a + b
        c = c * c
        del a
        del b
        del c
        torch.cuda.synchronize()
        tot_time += time.time_ns() - s

    print(f"With del = {(tot_time / iters) / 1000000}")

    tot_time = 0
    for i in range(iters):
        s = time.time_ns()
        a = torch.randn(2, 2048, 4096 // 1, requires_grad=True)
        b = torch.randn(2, 2048, 4096 // 1, requires_grad=True)
        c = a + b + a + b
        c = c * c
        torch.cuda.synchronize()
        tot_time += time.time_ns() - s

    print(f"With no del = {(tot_time / iters) / 1000000}")
