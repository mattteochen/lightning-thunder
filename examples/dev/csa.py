import torch
import torch.nn as nn
import thunder
from thunder.backend_optimizer.optimizer import benchmark_trace

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y

with torch.device('cuda'):
    num_heads = 8
    heads_per_dim = 64 * 2
    embed_dimension = num_heads * heads_per_dim
    dtype = torch.float16
    model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to(dtype)
    print(model)
    batch_size = 16
    max_sequence_len = 1024
    x = torch.randn(batch_size, max_sequence_len, embed_dimension, dtype=dtype, requires_grad=True)

    jmodel_def = thunder.jit(model, executors=['torchcompile', 'nvfuser'])
    jmodel_auto = thunder.jit(model, autotune_type='runtime')

    warm_up_iters = 2
    iters = 10
    stream = torch.cuda.current_stream()

    for _ in range(warm_up_iters):
        y = jmodel_auto(x)
        yy = jmodel_def(x)
        yyy = model(x)
        torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
        torch.autograd.grad(yy, x, grad_outputs=torch.ones_like(y))

    print('deviation auto:', (jmodel_auto(x) - model(x)).abs().max().item())
    print('deviation def:', (jmodel_def(x) - model(x)).abs().max().item())

    print('\n\n')

    for i in range(1):

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        middle_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record(stream)
            y = jmodel_auto(x)
            middle_events[i].record(stream)
            torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

        torch.cuda.synchronize()
        fw = [s.elapsed_time(e) for s, e in zip(start_events, middle_events)]
        bw = [s.elapsed_time(e) for s, e in zip(middle_events, end_events)]
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        fw_time = sum(fw)
        bw_time = sum(bw)
        tot_time = sum(tot)
        print(f'Auto fw: {fw_time / iters}')
        print(f'Auto bw: {bw_time / iters}')
        print(f'Auto tot: {tot_time / iters}')
        print('\n')

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        middle_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            torch.cuda.empty_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record(stream)
            y = jmodel_def(x)
            middle_events[i].record(stream)
            torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))
            end_events[i].record(stream)

        torch.cuda.synchronize()
        fw = [s.elapsed_time(e) for s, e in zip(start_events, middle_events)]
        bw = [s.elapsed_time(e) for s, e in zip(middle_events, end_events)]
        tot = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        fw_time = sum(fw)
        bw_time = sum(bw)
        tot_time = sum(tot)
        print(f'Default fw: {fw_time / iters}')
        print(f'Default bw: {bw_time / iters}')
        print(f'Default tot: {tot_time / iters}')
        print('-------------------------------------------------------')

        c, m, o = benchmark_trace(thunder.last_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='def_fw')
        print(f'Executing default fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='auto_fw')
        print(f'Executing auto fw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_def)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='def_bw')
        print(f'Executing default bw trace:\n{c} ms, {m / (2**30)} GB')
        del o
        c, m, o = benchmark_trace(thunder.last_backward_traces(jmodel_auto)[-1], apply_del_last_used=False, snapshot=True, snapshot_name='auto_bw')
        print(f'Executing auto bw trace:\n{c} ms, {m / (2**30)} GB')
        del o

    print('\n\n\n\n\n\n')
    print(f'{thunder.last_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_traces(jmodel_auto)[-1]}')

    print('\n\n')
    print(f'{thunder.last_backward_traces(jmodel_def)[-1]}')
    print('###############################################################################')
    print(f'{thunder.last_backward_traces(jmodel_auto)[-1]}')

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("def"):
            y = jmodel_def(x)
            grad_outputs = torch.ones_like(y)
            torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("auto"):
            y = jmodel_auto(x)
            grad_outputs = torch.ones_like(y)
            torch.autograd.grad(y, x, grad_outputs=grad_outputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

