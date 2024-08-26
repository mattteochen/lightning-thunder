from typing import Sequence
import thunder.backend_optimizer.utils as aut_utils
import pytest
import torch
import thunder
from thunder.core.prims import PrimIDs
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx
from thunder.extend import Executor, get_always_executors
from thunder.executors.torchex import ex as torchex
from thunder.executors.torch_compile import torch_compile_ex
from thunder.executors.nvfuserex import nvfuserex
from thunder.tests.framework import requiresCUDA, run_snippet

class DummyProxy():
    def __init__(self, name) -> None:
        self.name = name

@pytest.mark.parametrize("data,expected",
    [
        ([DummyProxy('a'), DummyProxy('b')], '[a#b#]'),
        ([DummyProxy('a'), DummyProxy('b'), 90], '[a#b#int90#]'),
        ([DummyProxy('a'), DummyProxy('b'), 90, None], '[a#b#int90#None#]'),
        ([DummyProxy('a'), DummyProxy('b'), 90, [DummyProxy('c'), [DummyProxy('d')]]], '[a#b#int90#[c#[d#]]]')
    ]
)
def test_sequence_hash(data, expected):
    assert(aut_utils.sequence_hash(data) == expected)

@pytest.mark.parametrize("data,expected",
    [
        ([DummyProxy('a'), "b"], '[a#b#]'),
    ]
)
def test_sequence_hash_bad_input(data, expected):
    with pytest.raises(AssertionError):
        assert aut_utils.sequence_hash(data) == expected


@pytest.mark.parametrize("data,expected_sum,expected_others",
    [
        ([nvfuserex, torch_compile_ex], Executor(name='empty'), Executor(name='empty')),
        ([nvfuserex, torchex], torchex, Executor(name='empty')),
    ]
)
def test_first_available_operator_executor(data,expected_sum,expected_others):
    def fn(a: torch.Tensor, b: torch.Tensor):
        return a + b

    a = torch.randn(1,1)
    b = torch.randn(1,1)
    jitted = thunder.jit(fn)
    jitted(a, b)
    trace = thunder.last_traces(jitted)[-1]
    for bsym in trace.bound_symbols:
        if bsym.sym.id == PrimIDs.ADD:
            assert aut_utils.get_first_available_operator_executor(bsym = bsym, executors = data, empty_hash = 'empty') == expected_sum
        else:
            assert aut_utils.get_first_available_operator_executor(bsym = bsym, executors = data, empty_hash = 'empty') == expected_others

@pytest.mark.parametrize("test,expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, [3, 4]], [1, 2, 3, 4]),
        ([1, 2, [3, 4, [None]]], [1, 2, 3, 4]),
    ]
)
def test_flatten_sequence(test, expected):
    assert aut_utils.flatten_sequence(test) == expected


def test_get_not_used_intermediate_outputs():
    # Flat outputs
    def fn(a: torch.Tensor, b: torch.Tensor):
        t1 = a - b
        t2 = a * b
        t3 = a / b
        return (a + b) + t2

    a = torch.randn(1,1)
    b = torch.randn(1,1)
    jitted = thunder.jit(fn, disable_dce=True)
    jitted(a, b)
    trace = thunder.last_traces(jitted)[-1]

    not_used = aut_utils.get_not_used_intermediate_outsputs(trace)
    # We have not used t1, t3 in trace
    not_used_labels = ['t1', 't3']
    assert len(not_used) == 2
    for t in not_used:
        assert t.name in not_used_labels
        not_used_labels.remove(t.name)

def _assign_executors_fn(a: torch.Tensor):
    t0 = a * 2
    t1 = a * a
    t3 = t0 + t1
    return t3

@pytest.mark.parametrize(
    "fn, args, executors",
    [
        (
            _assign_executors_fn,
            torch.randn(1, 1),
            [Executor("empty"), torchex, torchex, torchex, Executor("empty")],
        ),
        (
            _assign_executors_fn,
            torch.randn(1, 1),
            [Executor("empty"), torch_compile_ex, torch_compile_ex, torch_compile_ex, Executor("empty")],
        ),
        (
            _assign_executors_fn,
            torch.randn(1, 1),
            [Executor("empty"), torch_compile_ex, torch_compile_ex, torchex, Executor("empty")],
        ),
        (
            _assign_executors_fn,
            torch.randn(1, 1),
            [Executor("empty"), torchex, torch_compile_ex, torch_compile_ex, Executor("empty")],
        )
    ],
)
def test_assign_executors(fn, args, executors):
    trace: TraceCtx = thunder.trace(inline_trace=True)(fn, args)
    placed: TraceCtx = aut_utils.assign_executors(
        in_trace=trace, executors_list=executors, always_executors=get_always_executors(), empty_str="empty"
    )

    def _id(bsym: BoundSymbol):
        res = bsym.sym.name
        if isinstance(bsym.output, Sequence):
            res += '#' + aut_utils.sequence_hash(bsym.output)
        else:
            res += '#' + bsym.output.name

        return res

    # Unapacks and return symbols are filtered out
    executor_map = {
        _id(b): e if e.name != "empty" else None
        for b, e in zip(trace.bound_symbols, executors)
        if b.output is not None and b.sym.id != PrimIDs.RETURN
        and b.args is not None
    }

    for b in placed.bound_symbols:
        # print(b)
        if b.sym.is_fusion:
            # Search in every subsymbol
            for sub in b.subsymbols:
                identif = _id(sub)
                assert b.sym.executor == executor_map[identif]
        elif b.sym.id != PrimIDs.RETURN and b.args:
            identif = _id(b)
            assert b.sym.executor == executor_map[identif]

class Linear(torch.nn.Module):
    def __init__(self, a, b) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(a, b)

    def forward(self, x):
        return self.linear(x)

class Matmul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x @ x

@pytest.mark.parametrize(
    "model, x, op, expected",
    [
        (
            Linear(8,8),
            torch.randn(8,8),
            'linear',
            True
        ),
        (
            Linear(8,8),
            torch.randn(8,8),
            'add',
            False
        ),
        (
            Matmul(),
            torch.randn(8,8),
            'matmul',
            True
        ),
    ],
)
def test_operation_in_trace(model, x, op, expected):
    jitted = thunder.jit(model)
    jitted(x)
    # jitted(args if not isinstance(args, Sequence) else *args)
    trace = thunder.last_traces(jitted)[-1]

    assert aut_utils.operation_in_trace(trace=trace, op=op) == expected

class Sdpa(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_sdpa(device: str):
    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], dtype=torch.float16, device=device, requires_grad=True)
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], dtype=torch.float16, device=device, requires_grad=True)
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], dtype=torch.float16, device=device, requires_grad=True)

    model = Sdpa()
    executors = ['cudnn', 'sdpa']
    jitted = thunder.jit(model, autotune_type='runtime', executors=executors)
    jitted(query, key, value)

    exs: Sequence[Executor] = thunder.executors_applied(jitted)

    sdpa_executors_occurences = 0
    for ex in exs:
        if ex.name in executors:
            sdpa_executors_occurences += 1

    assert sdpa_executors_occurences == 1

