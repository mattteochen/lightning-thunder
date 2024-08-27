from typing import Callable, Sequence
import thunder.backend_optimizer.utils as aut_utils
import pytest
import torch
import thunder
from thunder.core.dtypes import to_torch_dtype
from thunder.core.prims import PrimIDs
from thunder.core.proxies import FloatProxy, IntegerProxy, TensorProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx
from thunder.extend import Executor, get_always_executors
from thunder.executors.torchex import ex as torchex
from thunder.executors.torch_compile import torch_compile_ex
from thunder.executors.nvfuserex import nvfuserex
from thunder.tests.framework import requiresCUDA


class DummyProxy:
    def __init__(self, name) -> None:
        self.name = name


@pytest.mark.parametrize(
    "data,expected",
    [
        ([DummyProxy("a"), DummyProxy("b")], "[a#b#]"),
        ([DummyProxy("a"), DummyProxy("b"), 90], "[a#b#int90#]"),
        ([DummyProxy("a"), DummyProxy("b"), 90, None], "[a#b#int90#None#]"),
        ([DummyProxy("a"), DummyProxy("b"), 90, [DummyProxy("c"), [DummyProxy("d")]]], "[a#b#int90#[c#[d#]]]"),
    ],
)
def test_sequence_hash(data, expected):
    assert aut_utils.sequence_hash(data) == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        ([DummyProxy("a"), "b"], "[a#b#]"),
    ],
)
def test_sequence_hash_bad_input(data, expected):
    with pytest.raises(AssertionError):
        assert aut_utils.sequence_hash(data) == expected


@pytest.mark.parametrize(
    "data,expected_sum,expected_others",
    [
        ([nvfuserex, torch_compile_ex], Executor(name="empty"), Executor(name="empty")),
        ([nvfuserex, torchex], torchex, Executor(name="empty")),
    ],
)
def test_first_available_operator_executor(data, expected_sum, expected_others):
    def fn(a: torch.Tensor, b: torch.Tensor):
        return a + b

    a = torch.randn(1, 1)
    b = torch.randn(1, 1)
    jitted = thunder.jit(fn)
    jitted(a, b)
    trace = thunder.last_traces(jitted)[-1]
    for bsym in trace.bound_symbols:
        if bsym.sym.id == PrimIDs.ADD:
            assert (
                aut_utils.get_first_available_operator_executor(bsym=bsym, executors=data, empty_hash="empty")
                == expected_sum
            )
        else:
            assert (
                aut_utils.get_first_available_operator_executor(bsym=bsym, executors=data, empty_hash="empty")
                == expected_others
            )


@pytest.mark.parametrize(
    "test,expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, [3, 4]], [1, 2, 3, 4]),
        ([1, 2, [3, 4, [None]]], [1, 2, 3, 4]),
    ],
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

    a = torch.randn(1, 1)
    b = torch.randn(1, 1)
    jitted = thunder.jit(fn, disable_dce=True)
    jitted(a, b)
    trace = thunder.last_traces(jitted)[-1]

    not_used = aut_utils.get_not_used_intermediate_outsputs(trace)
    # We have not used t1, t3 in trace
    not_used_labels = ["t1", "t3"]
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
        ),
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
            res += "#" + aut_utils.sequence_hash(bsym.output)
        else:
            res += "#" + bsym.output.name

        return res

    # Unapacks and return symbols are filtered out
    executor_map = {
        _id(b): e if e.name != "empty" else None
        for b, e in zip(trace.bound_symbols, executors)
        if b.output is not None and b.sym.id != PrimIDs.RETURN and b.args is not None
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
        (Linear(8, 8), torch.randn(8, 8), "linear", True),
        (Linear(8, 8), torch.randn(8, 8), "add", False),
        (Matmul(), torch.randn(8, 8), "matmul", True),
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


@pytest.mark.parametrize(
    "model, q, k, v, executors, expected",
    [
        (
            Sdpa(),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            ["cudnn", "sdpa", "fa3"],
            1,
        ),
        (
            Sdpa(),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            ["cudnn", "sdpa"],
            1,
        ),
        (
            Sdpa(),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            [
                "cudnn",
            ],
            1,
        ),
        (
            Sdpa(),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            torch.randn([10, 128, 4, 32], dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True),
            [],
            0,
        ),
    ],
)
@requiresCUDA
# Currently these executors are: cudnn, spda, fa3, TE
def test_update_compile_options_executor_list_after_fw_bw_split(model, q, k, v, executors, expected):
    jitted = thunder.jit(model, autotune_type="runtime", executors=executors)
    jitted(q, k, v)

    assigned: Sequence[Executor] = thunder.executors_applied(jitted)

    count = 0
    for ex in assigned:
        count += 1 if ex.name in executors else 0

    assert count == expected


def _test_transform_proxy_to_torch_fn_1(a: torch.Tensor, b: torch.Tensor, k: int):
    t0 = a * b
    return t0 * k


def _test_transform_proxy_to_torch_fn_2(
    a: torch.Tensor, b: torch.Tensor, c: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
):
    t0 = c[0] + c[1][0]
    t1 = t0 * c[1][1]
    return t1 - a + b


def _test_transform_proxy_to_torch_common(
    fn: Callable, torch_args: tuple, executors: list, has_backward: bool, **kwargs
):
    jitted = thunder.jit(fn, executors=executors)
    jitted(*torch_args)

    trace_static_args = thunder.last_traces(jitted)[-1].args
    assert trace_static_args

    transformed_args = aut_utils.transform_proxy_to_torch(trace_static_args, **kwargs)

    assert isinstance(transformed_args, list)

    def _comp(thunder_seq: Sequence, torch_seq: Sequence):
        assert len(thunder_seq) == len(torch_seq)

        for a, b in zip(thunder_seq, torch_seq):
            if isinstance(a, TensorProxy):
                # handle TE fp32
                # Static type for fp8 is torch.float8 but the runtime is TE Float8 if TE is being used
                if a.dtype.bytes == 1 and kwargs.get("te_used"):
                    assert b.dtype == torch.float32
                else:
                    assert b.dtype == to_torch_dtype(a.dtype)
                assert a.device.device_str() == str(b.device)
                assert a.shape == b.shape
                assert a.requires_grad == b.requires_grad
            elif isinstance(a, IntegerProxy) or isinstance(a, FloatProxy):
                assert a.value == b

            if isinstance(a, Sequence):
                assert isinstance(b, Sequence)
                _comp(a, b)

    _comp(trace_static_args, transformed_args)

    if has_backward:
        trace_static_args = thunder.last_backward_traces(jitted)[-1].args
        assert trace_static_args

        transformed_args = aut_utils.transform_proxy_to_torch(trace_static_args, **kwargs)
        print(trace_static_args)
        # print(transformed_args)

        _comp(trace_static_args, transformed_args)


@pytest.mark.parametrize(
    "fn, torch_args, executors, has_backward",
    [
        (_test_transform_proxy_to_torch_fn_1, tuple([torch.randn(1, 1), torch.randn(1, 1), 10]), [], False),
        (
            _test_transform_proxy_to_torch_fn_2,
            tuple([torch.randn(1, 1), torch.randn(1, 1), (torch.randn(1, 1), (torch.randn(1, 1), torch.rand(1, 1)))]),
            [],
            False,
        ),
        (
            Sdpa(),
            (
                torch.randn([10, 128, 4, 32], dtype=torch.float16, requires_grad=True),
                torch.randn([10, 128, 4, 32], dtype=torch.float16, requires_grad=True),
                torch.randn([10, 128, 4, 32], dtype=torch.float16, requires_grad=True),
            ),
            [],
            True,
        ),
    ],
)
def test_transform_proxy_to_torch(fn: Callable, torch_args: tuple, executors: list, has_backward: bool):
    _test_transform_proxy_to_torch_common(fn, torch_args, executors, has_backward)


@requiresCUDA
def test_transform_proxy_to_torch_TE():
    class Model(torch.nn.Module):
        def __init__(self, in_features, out_features) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = Model(4096, 4096)
    model.to("cuda")

    _test_transform_proxy_to_torch_common(
        model,
        tuple([torch.randn(4096, 4096, requires_grad=True, device="cuda")]),
        ["transformer_engine"],
        True,
        te_used=True,
    )


@pytest.mark.parametrize(
    "executors, expected",
    [
        (["python"], ["nvfuser", "python"]),
        (["nvfuser", "cudnn"], ["cudnn", "nvfuser"]),
        (["torch", "nvfuser", "sdpa"], ["sdpa", "torch", "nvfuser"]),
        (["transformer_engine", "nvfuser", "sdpa"], ["transformer_engine", "sdpa", "nvfuser"]),
    ],
)
def test_reorder_executors_list(executors, expected):
    assert aut_utils.reorder_executors_list(executors) == expected
