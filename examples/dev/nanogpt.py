import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark
from thunder.tests.nanogpt_model import GPTConfig, GPT
from contextlib import nullcontext
import time
# import os

warm_up_iters = 50


def bench():
    # -----------------------------------------------------------------------------
    batch_size = 12
    block_size = 1024
    bias = False
    real_data = True
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    profile = False # use pytorch profiler, or just simple benchmarking?
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # data loading init
    if real_data:
        raise RuntimeError('Not supported')
        # dataset = 'openwebtext'
        # data_dir = os.path.join('data', dataset)
        # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        # def get_batch(split):
        #     data = train_data # note ignore split in benchmarking script
        #     ix = torch.randint(len(data) - block_size, (batch_size,))
        #     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        #     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        #     return x, y
    else:
        # alternatively, if fixed data is desired to not care about data loading
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

    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

    if compile:
        print("Compiling model...")
        model = torch.compile(model) # pytorch 2.0

    if profile:
        # useful docs on pytorch profiler:
        # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
        # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:

            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch('train')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")

                prof.step() # notify the profiler at end of each step

    else:

        # simple benchmarking
        torch.cuda.synchronize()
        for stage, num_steps in enumerate([50, 50]): # burnin, then benchmark
            t0 = time.time()
            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = jmodel_def(X, Y)
                X, Y = get_batch('train')
                # optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1-t0
            # mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
            if stage == 1:
                # print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
                print(f"time per iteration default model: {dt/num_steps*1000:.4f}ms")

        # simple benchmarking
        torch.cuda.synchronize()
        for stage, num_steps in enumerate([50, 50]): # burnin, then benchmark
            t0 = time.time()
            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = jmodel_auto(X, Y)
                X, Y = get_batch('train')
                # optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1-t0
            # mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
            if stage == 1:
                # print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
                print(f"time per iteration auto model: {dt/num_steps*1000:.4f}ms")


        print('\n\nResults thunder benchmark:')
        traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
        labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
        thunder_fw_bw_benchmark(traces, labels, 50)

# def torch_fw_bw_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

#     for m, input, label in zip(models, inputs, labels):
#         # Warm up
#         for _ in range(warm_up_iters):
#             _, loss = m(input)
#             loss.backward()

#         torch.cuda.synchronize()
#         start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         stream = torch.cuda.current_stream()
#         max_allocated_bytes = 0
#         for i in range(iters):
#             torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
#             torch.cuda.empty_cache()
#             torch.cuda._sleep(1_000_000)

#             start_events[i].record(stream)
#             out = m(input)
#             end_events[i].record(stream)

#             max_allocated_bytes = max(
#                 max_allocated_bytes, torch.cuda.max_memory_allocated(
#                     torch.cuda.current_device())
#             )

#         torch.cuda.synchronize()
#         tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
#         tot_time = sum(tot) / iters
#         print(f'{label} tot fw time: {tot_time} ms')
#         print(f'{label} max fw allocated memory: {max_allocated_bytes / (2**30)} GB')

#         torch.cuda.synchronize()
#         start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         stream = torch.cuda.current_stream()
#         max_allocated_bytes = 0
#         for i in range(iters):
#             torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
#             torch.cuda.empty_cache()
#             torch.cuda._sleep(1_000_000)

#             _, loss = m(input)
#             loss.backward()
#             start_events[i].record(stream)
#             loss.backward()
#             end_events[i].record(stream)

#             max_allocated_bytes = max(
#                 max_allocated_bytes, torch.cuda.max_memory_allocated(
#                     torch.cuda.current_device())
#             )

#         torch.cuda.synchronize()
#         tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
#         tot_time = sum(tot) / iters
#         print(f'{label} tot bw time: {tot_time} ms')
#         print(f'{label} max bw allocated memory: {max_allocated_bytes / (2**30)} GB')

# def torch_total_benchmark(models: list, labels: list, inputs: list, iters: int) -> None:

#     for m, input, label in zip(models, inputs, labels):
#         # Warm up
#         for _ in range(warm_up_iters):
#             _, loss = m(input)
#             loss.backward()

#         start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
#         torch.cuda.synchronize()
#         stream = torch.cuda.current_stream()
#         max_allocated_bytes = 0
#         for i in range(iters):
#             torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
#             torch.cuda.empty_cache()
#             torch.cuda._sleep(1_000_000)

#             start_events[i].record(stream)
#             _, loss = m(input)
#             loss.backward()
#             end_events[i].record(stream)

#             max_allocated_bytes = max(
#                 max_allocated_bytes, torch.cuda.max_memory_allocated(
#                     torch.cuda.current_device())
#             )

#         torch.cuda.synchronize()
#         tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
#         tot_time = sum(tot) / iters
#         print(f'{label} tot time: {tot_time} ms')
#         print(f'{label} max allocated memory: {max_allocated_bytes / (2**30)} GB')

# # -----------------------------------------------------------------------------
# batch_size = 12
# block_size = 1024
# bias = False
# seed = 1337
# device = 'cuda'
# # dtype = 'float16'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# # -----------------------------------------------------------------------------
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# x = torch.randint(50304, (batch_size, block_size), device=device)
# y = torch.randint(50304, (batch_size, block_size), device=device)
# get_batch = lambda split: (x, y)

# # model init
# gptconf = GPTConfig(
#     block_size = block_size, # how far back does the model look? i.e. context size
#     n_layer = 1, n_head = 12, n_embd = 768, # size of the model
#     dropout = 0, # for determinism
#     bias = bias,
# )
# model = GPT(gptconf)
# model.to(device)

# jmodel_def = thunder.jit(model)
# jmodel_auto = thunder.jit(model, autotune_type='runtime', executors = ['nvfuser', 'torchcompile', 'sdpa', 'cudnn', 'torch', 'python'])

# X, Y = get_batch('train')
# # Run compilation
# jmodel_def(x, y)
# jmodel_auto(x, y)

# print('Results thunder benchmark:')
# traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
# labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
# thunder_fw_bw_benchmark(traces, labels, 50)

# print('\n\nResults torch fw bw benchmark:')
# callables = [jmodel_def, jmodel_auto]
# labels = ['def', 'auto']
# inputs = [x, x]
# torch_fw_bw_benchmark(callables, labels, inputs, 50)
# print('\n\nResults torch tot benchmark:')
# torch_total_benchmark(callables, labels, inputs, 50)

# print('\n\n\n\n\n\n')
# print(f'{thunder.last_traces(jmodel_def)[-1]}')
# print('###############################################################################')
# print(f'{thunder.last_traces(jmodel_auto)[-1]}')

# print('\n\n')
# print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
# print('###############################################################################')
# print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

bench()
