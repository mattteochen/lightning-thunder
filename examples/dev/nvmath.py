# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of torch tensors.

nvmath-python supports multiple frameworks. The result of each operation is a tensor of the same
framework that was used to pass the inputs. It is also located on the same device as the inputs.
"""

import torch
import nvmath
import thunder

torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

a = torch.randn(128256, 8192, device="cuda")
b = torch.randn(8192, 4096, device="cuda")

iters = 20
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
stream = torch.cuda.current_stream()
for i in range(iters):
    result_a = torch.matmul(a, b)
torch.cuda.default_stream().synchronize()
for i in range(iters):
    torch.cuda.empty_cache()
    torch.cuda._sleep(1_000_000)
    start_events[i].record(stream)
    result_a = torch.matmul(a, b)
    end_events[i].record(stream)
torch.cuda.default_stream().synchronize()
tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
tot_time = sum(tot) / iters
print(f"torch tot time: {tot_time} ms")

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
stream = torch.cuda.current_stream()
torch.cuda.default_stream().synchronize()
for i in range(iters):
    result_b = nvmath.linalg.advanced.matmul(a, b)
torch.cuda.default_stream().synchronize()
for i in range(iters):
    torch.cuda.empty_cache()
    torch.cuda._sleep(1_000_000)
    start_events[i].record(stream)
    result_b = nvmath.linalg.advanced.matmul(a, b)
    end_events[i].record(stream)
torch.cuda.default_stream().synchronize()
tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
tot_time = sum(tot) / iters
print(f"nvmath tot time: {tot_time} ms")

with nvmath.linalg.advanced.Matmul(a, b) as mm:
    # Plan.
    mm.plan()

    # Inspect the algorithms proposed.
    print(
        f"Planning returned {len(mm.algorithms)} algorithms. The capabilities of the best one are:",
    )
    best = mm.algorithms[0]
    print(best.capabilities)

    # Modify the tiling configuration of the algorithm. Note that the valid tile configuration depends on
    # the hardware, and not all combinations of the configuration are supported, so we leave it as an exercise.
    print(f"Tiling {best.tile}")
    # Execute the multiplication.

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stream = torch.cuda.current_stream()
    torch.cuda.default_stream().synchronize()
    for i in range(iters):
        result_c = mm.execute()
    torch.cuda.default_stream().synchronize()
    for i in range(iters):
        torch.cuda.empty_cache()
        torch.cuda._sleep(1_000_000)
        start_events[i].record(stream)
        result_c = mm.execute()
        end_events[i].record(stream)
    torch.cuda.default_stream().synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    tot_time = sum(tot) / iters
    print(f"nvmath tuned tot time: {tot_time} ms")

with nvmath.linalg.advanced.Matmul(a, b) as mm:
    # Plan.
    mm.plan()

    # Autotune
    mm.autotune(iterations=5)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stream = torch.cuda.current_stream()
    torch.cuda.default_stream().synchronize()
    for i in range(iters):
        result_d = mm.execute()
    torch.cuda.default_stream().synchronize()
    for i in range(iters):
        torch.cuda.empty_cache()
        torch.cuda._sleep(1_000_000)
        start_events[i].record(stream)
        result_d = mm.execute()
        end_events[i].record(stream)
    torch.cuda.default_stream().synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    tot_time = sum(tot) / iters
    print(f"nvmath tuned tot time: {tot_time} ms")


print("deviation:", (result_a - result_b).abs().max().item())
print("deviation:", (result_a - result_c).abs().max().item())
print("deviation:", (result_a - result_d).abs().max().item())

# from thunder.benchmarks.utils import thunder_fw_bw_benchmark
# from thunder.executors.nvmathex import nvmath_ex

# class Module(torch.nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def forward(self, a, b):
#         return a @ b

# with torch.device('cuda'):

#     model = Module()
#     jmodel_def = thunder.jit(model)
#     jmodel_auto = thunder.jit(model, autotune_type="runtime", executors = [nvmath_ex], use_cudagraphs=False)
#     a = torch.randn(128256, 128, requires_grad=True)
#     b = torch.randn(128, 4096, requires_grad=True)

#     print('deviation def:', (jmodel_def(a, b) - model(a, b)).abs().max().item())
#     print('deviation auto:', (jmodel_auto(a, b) - model(a, b)).abs().max().item())

#     from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_fw_bw_benchmark, torch_fw_bw_benchmark_nvsight, torch_total_benchmark

#     print('Results with thunder benchmark:')
#     fw_traces = [
#         thunder.last_traces(jmodel_def)[-1],
#         thunder.last_traces(jmodel_auto)[-1],
#     ]
#     bw_traces = [
#         thunder.last_backward_traces(jmodel_def)[-1],
#         thunder.last_backward_traces(jmodel_auto)[-1],
#     ]
#     fw_labels = ["fw_def", "fw_auto"]
#     bw_labels = ["bw_def", "bw_auto"]
#     thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, 10)

#     for t in fw_traces:
#         print(f'{t}\n################################')
#     for t in bw_traces:
#         print(f'{t}\n################################')

# for _ in range(iters):
#     result_a = model(a.clone().detach(), b.clone().detach())
# torch.cuda.default_stream().synchronize()
# s = time.time_ns()
# for i in range(iters):
#     result_a = model(a.clone().detach(), b.clone().detach())
# torch.cuda.default_stream().synchronize()
# e = time.time_ns()
# print('time torch', (e-s)/1000000, 'ms')

# for _ in range(iters):
#     result_b = jmodel(a.clone().detach(), b.clone().detach())
# torch.cuda.default_stream().synchronize()
# s = time.time_ns()
# for i in range(iters):
#     result_b = model(a.clone().detach(), b.clone().detach())
# torch.cuda.default_stream().synchronize()
# e = time.time_ns()
# print('time nvmath', (e-s)/1000000, 'ms')
