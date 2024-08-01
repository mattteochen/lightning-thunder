import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark
from thunder.tests.nanogpt_model import GPTConfig, GPT

warm_up_iters = 50

def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            _, loss = m(input)
            loss.backward()

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            start_events[i].record(stream)
            out = m(input)
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot fw time: {tot_time} ms')
        print(f'{label} max fw allocated memory: {max_allocated_bytes / (2**30)} GB')

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            _, loss = m(input)
            loss.backward()
            start_events[i].record(stream)
            loss.backward()
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot bw time: {tot_time} ms')
        print(f'{label} max bw allocated memory: {max_allocated_bytes / (2**30)} GB')

def torch_total_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

    for m, input, label in zip(models, inputs, labels):
        # Warm up
        for _ in range(warm_up_iters):
            _, loss = m(input)
            loss.backward()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream()
        max_allocated_bytes = 0
        for i in range(iters):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)

            start_events[i].record(stream)
            _, loss = m(input)
            loss.backward()
            end_events[i].record(stream)

            max_allocated_bytes = max(
                max_allocated_bytes, torch.cuda.max_memory_allocated(
                    torch.cuda.current_device())
            )

        torch.cuda.synchronize()
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        tot_time = sum(tot) / iters
        print(f'{label} tot time: {tot_time} ms')
        print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} GB')

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
seed = 1337
device = 'cuda'
# dtype = 'float16'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

x = torch.randint(50304, (batch_size, block_size), device=device)
y = torch.randint(50304, (batch_size, block_size), device=device)
get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 1, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

jmodel_def = thunder.jit(model)
jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])

X, Y = get_batch('train')
# Run compilation
jmodel_def(x, y)
jmodel_auto(x, y)

print('Results thunder benchmark:')
traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
thunder_fw_bw_benchmark(traces, labels, 50)

print('\n\nResults torch fw bw benchmark:')
callables = [jmodel_def, jmodel_auto]
labels = ['def', 'auto']
inputs = [x, x]
torch_fw_bw_benchmark(callables, labels, inputs, 50)
print('\n\nResults torch tot benchmark:')
torch_total_benchmark(callables, labels, inputs, 50)

print('\n\n\n\n\n\n')
print(f'{thunder.last_traces(jmodel_def)[-1]}')
print('###############################################################################')
print(f'{thunder.last_traces(jmodel_auto)[-1]}')

print('\n\n')
print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
print('###############################################################################')
print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

