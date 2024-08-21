import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark, torch_total_benchmark

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.set_default_dtype(dtype)
print(f'Script data type: {dtype}\n')

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query, key, value):
        a = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        # Make different inputs as happens in a real model
        b = torch.nn.functional.scaled_dot_product_attention(query+query, key+key, value+value)
        # c = torch.nn.functional.scaled_dot_product_attention(query*query, key*key, value*value)
        # d = torch.nn.functional.scaled_dot_product_attention(query-query, key-key, value-value)
        return a + b


def bench(m, label, iters):
    q = torch.rand(32, 8, 128, 64*1, requires_grad=True)
    k = torch.rand(32, 8, 128, 64*1, requires_grad=True)
    v = torch.rand(32, 8, 128, 64*1, requires_grad=True)

    # warm up
    for _ in range(50):
        y = m(q, k, v)
        # y.sum().backward()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stream = torch.cuda.current_stream()
    max_allocated_bytes = 0
    torch.cuda.synchronize()
    for i in range(iters):
        torch.cuda.empty_cache()
        torch.cuda._sleep(1_000_000)
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

        start_events[i].record(stream)
        y = m(q, k, v)
        loss = y.sum()
        # loss.backward()
        end_events[i].record(stream)

        max_allocated_bytes = max(
            max_allocated_bytes, torch.cuda.max_memory_allocated(
                torch.cuda.current_device())
        )

    torch.cuda.synchronize()
    tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    tot_time = sum(tot) / iters
    print(f'{label} tot time: {tot_time} ms')
    print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} gb')

with torch.device('cuda'):
    model = Model()

    jmodel_def = thunder.jit(model)
    jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'cudnn', 'sdpa'], use_cudagraphs=False)

    q = torch.rand(32, 8, 128, 64*1, requires_grad=True)
    k = torch.rand(32, 8, 128, 64*1, requires_grad=True)
    v = torch.rand(32, 8, 128, 64*1, requires_grad=True)

    jmodel_def(q, k, v)
    jmodel_auto(q, k, v)

    iters = 100
    fw_traces = [
        thunder.last_traces(jmodel_def)[-1],
        thunder.last_traces(jmodel_auto)[-1],
    ]
    bw_traces = [
        thunder.last_backward_traces(jmodel_def)[-1],
        thunder.last_backward_traces(jmodel_auto)[-1],
    ]
    fw_labels = ["fw_def", "fw_auto"]
    bw_labels = ["bw_def", "bw_auto"]
    print('Thunder benchmark:')
    thunder_fw_bw_benchmark(fw_traces, bw_traces, fw_labels, bw_labels, iters)


    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

    print('\nTorch benchmark:')
    bench(jmodel_def, 'def', iters)
    bench(jmodel_auto, 'auto', iters)
