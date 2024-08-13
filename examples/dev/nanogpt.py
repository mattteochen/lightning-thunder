import torch
import thunder
from thunder.benchmarks.utils import thunder_fw_bw_benchmark
from thunder.tests.nanogpt_model import GPTConfig, GPT
from contextlib import nullcontext

warm_up_iters = 50

def run(target: str = 'runtime'):
    if target != 'runtime' and target != 'memory':
        raise AssertionError(f'Target {target} not supported. Only runtime and memory available')
    # -----------------------------------------------------------------------------
    batch_size = 12
    block_size = 512
    bias = False
    real_data = False
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    profile = False # use pytorch profiler, or just simple benchmarking?
    # exec(open('configurator.py').read()) # overrides from command line or config file
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
    else:
        # alternatively, if fixed data is desired to not care about data loading
        x = torch.randint(50304, (batch_size, block_size), device=device)
        y = torch.randint(50304, (batch_size, block_size), device=device)
        get_batch = lambda split: (x, y)

    # model init
    gptconf = GPTConfig(
        block_size = block_size, # how far back does the model look? i.e. context size
        n_layer = 1, n_head = 6, n_embd = 768, # size of the model
        dropout = 0, # for determinism
        bias = bias,
    )
    model = GPT(gptconf)
    model.to(device)

    jmodel_def = thunder.jit(model, use_cudagraphs=True)
    # Currently sdpa does not work?
    jmodel_auto = thunder.jit(model, autotune_type=target, executors = ['torchcompile', 'nvfuser', 'cudnn', 'torch', 'python'], use_cudagraphs=True)

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

            models = [jmodel_def, jmodel_auto]

            for mod in models:
                print('Profiling new model')
                X, Y = get_batch('train')
                for k in range(num_steps):
                    with ctx:
                        _, loss = mod(X, Y)
                    X, Y = get_batch('train')
                    loss.backward()
                    lossf = loss.item()
                    print(f"{k}/{num_steps} loss: {lossf:.4f}")

                    prof.step() # notify the profiler at end of each step

    else:
        def measure(m, label):
            # simple benchmarking
            torch.cuda.synchronize()

            X, Y = get_batch('train')
            for i in range(warm_up_iters):
                with ctx:
                    _, loss = m(X, Y)
                X, Y = get_batch('train')
                loss.backward()

            iters = 100
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            stream = torch.cuda.current_stream()
            torch.cuda.synchronize()
            X, Y = get_batch('train')
            for i in range(iters):
                torch.cuda.empty_cache()
                torch.cuda._sleep(1_000_000)
                start_events[i].record(stream)
                with ctx:
                    _, loss = m(X, Y)
                X, Y = get_batch('train')
                loss.backward()
                end_events[i].record(stream)

            torch.cuda.synchronize()
            tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            tot_time = sum(tot) / iters
            print('\n\nResults torch benchmark:')
            print(f'{label} tot time: {tot_time} ms')

        measure(jmodel_auto, 'auto')
        measure(jmodel_def, 'def')

        print('\n\nResults thunder benchmark:')
        traces = [
            thunder.last_traces(jmodel_def)[-1],
            thunder.last_traces(jmodel_auto)[-1],
            thunder.last_backward_traces(jmodel_def)[-1],
            thunder.last_backward_traces(jmodel_auto)[-1],
        ]
        traces.reverse()
        labels = ['fw_def', 'fw_auto', 'bw_def', 'bw_auto']
        labels.reverse()
        thunder_fw_bw_benchmark(traces, labels, 100)

        # X, Y = get_batch('train')
        # out_eager = model(X, Y)
        # out_def = jmodel_def(X, Y)
        # out_auto = jmodel_auto(X, Y)
        # for a, b in zip(out_eager, out_def):
        #     print('deviation def:', (a - b).abs().max().item())
        # for a, b in zip(out_eager, out_auto):
        #     print('deviation auto:', (a - b).abs().max().item())

    traces = [thunder.last_traces(jmodel_def)[-1], thunder.last_traces(jmodel_auto)[-1], thunder.last_backward_traces(jmodel_def)[-1], thunder.last_backward_traces(jmodel_auto)[-1]]
    for t in traces:
        print(f'{t}\n############################################')

run()
