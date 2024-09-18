from thunder.backend_optimizer.utils import get_fw_bw_split_backends_options
from thunder.core.dtypes import to_torch_dtype
from thunder.core.prims import PrimIDs
from thunder.core.proxies import FloatProxy, IntegerProxy, TensorProxy
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx
from thunder.executors.cudnnex import cudnn_ex
from thunder.executors.fa3ex import fa3_ex
from thunder.executors.nvfuserex import nvfuserex
from thunder.executors.pythonex import ex as pythonex
from thunder.executors.sdpaex import sdpa_ex
from thunder.executors.torch_compile import torch_compile_ex
from thunder.executors.torchex import ex as torchex
from thunder.executors.transformer_engineex import transformer_engine_ex
from thunder.extend import Executor, get_always_executors
from thunder.tests.framework import requiresCUDA
from typing import Callable, Sequence
import pytest
import thunder
import thunder.backend_optimizer.utils as aut_utils
import torch


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
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            ["cudnn", "sdpa", "fa3"],
            1,
        ),
        (
            Sdpa(),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            ["cudnn", "sdpa"],
            1,
        ),
        (
            Sdpa(),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            [
                "cudnn",
            ],
            1,
        ),
        (
            Sdpa(),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
            torch.randn(
                [10, 128, 4, 32],
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True,
            ),
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


def _test_transform_proxies_to_real_fn_1(a: torch.Tensor, b: torch.Tensor, k: int):
    t0 = a * b
    return t0 * k


def _test_transform_proxies_to_real_fn_2(
    a: torch.Tensor, b: torch.Tensor, c: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
):
    t0 = c[0] + c[1][0]
    t1 = t0 * c[1][1]
    return t1 - a + b


def _test_transform_proxies_to_real_common(
    fn: Callable, torch_args: tuple, executors: list, has_backward: bool, **kwargs
):
    jitted = thunder.jit(fn, executors=executors)
    jitted(*torch_args)

    trace_static_args = thunder.last_traces(jitted)[-1].args
    assert trace_static_args

    transformed_args = aut_utils.transform_proxies_to_real(trace_static_args, **kwargs)

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

        transformed_args = aut_utils.transform_proxies_to_real(trace_static_args, **kwargs)
        print(trace_static_args)
        # print(transformed_args)

        _comp(trace_static_args, transformed_args)


@pytest.mark.parametrize(
    "fn, torch_args, executors, has_backward",
    [
        (_test_transform_proxies_to_real_fn_1, tuple([torch.randn(1, 1), torch.randn(1, 1), 10]), [], False),
        (
            _test_transform_proxies_to_real_fn_2,
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
def test_transform_proxies_to_real(fn: Callable, torch_args: tuple, executors: list, has_backward: bool):
    _test_transform_proxies_to_real_common(fn, torch_args, executors, has_backward)


@requiresCUDA
def test_transform_proxies_to_real_TE():
    class Model(torch.nn.Module):
        def __init__(self, in_features, out_features) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = Model(4096, 4096)
    model.to("cuda")

    _test_transform_proxies_to_real_common(
        model,
        tuple([torch.randn(4096, 4096, requires_grad=True, device="cuda")]),
        ["transformer_engine"],
        True,
        te_used=True,
    )


@pytest.mark.parametrize(
    "executors, expected, use_te",
    [
        (["python"], ["nvfuser", "python"], False),
        (["nvfuser", "cudnn"], ["cudnn", "nvfuser"], False),
        (["torch", "nvfuser", "sdpa"], ["sdpa", "torch", "nvfuser"], False),
        (["transformer_engine", "nvfuser", "sdpa"], ["transformer_engine", "sdpa", "nvfuser"], True),
    ],
)
# We might not have nvfuser in non cuda envs
@requiresCUDA
def test_reorder_executors_list(executors, expected, use_te):
    assert aut_utils.reorder_executors_list(executors, autotune_enable_te=use_te) == expected


@pytest.mark.parametrize(
    "name, expected",
    [("linear", [transformer_engine_ex]), ("scaled_dot_product_attention", [sdpa_ex, cudnn_ex, fa3_ex])],
)
def test_get_fw_bw_split_backends_options(name: str, expected):
    symbol = Symbol(name=name)
    bsym = BoundSymbol(symbol, (), {}, None)
    options = get_fw_bw_split_backends_options(bsym, autotune_enable_te=True)
    assert all(map(lambda v: v in options, expected))


class Model_1(torch.nn.Module):
    def __init__(self, in_f, out_f) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        t0 = self.linear(x)
        return torch.nn.functional.silu(t0)


class Model_2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_head = 12
        self.n_embd = 3072
        self.c_attn = torch.nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize(
    "model, tensor_shape, dtype, autotune_type, executors, expected_executors, use_te",
    [
        (
            Model_1(32, 32),
            (32, 32),
            torch.float32,
            "runtime",
            [nvfuserex],
            [[nvfuserex, torchex, pythonex]],
            False,
        ),
        (
            Model_1(32, 32),
            (32, 32),
            torch.float32,
            "memory",
            [torch_compile_ex],
            [[torch_compile_ex, torchex, pythonex]],
            False,
        ),
        (
            Model_1(4096, 4096),
            (128, 4096),
            torch.float32,
            "runtime",
            [transformer_engine_ex],
            [[transformer_engine_ex, nvfuserex, torchex, pythonex]],
            True,
        ),
        (
            Model_2(),
            (16, 1024, 3072),
            torch.float16,
            "runtime",
            [sdpa_ex, cudnn_ex],
            [[sdpa_ex, nvfuserex, torchex, pythonex], [cudnn_ex, nvfuserex, torchex, pythonex]],
            False,
        ),
        (
            Model_2(),
            (16, 1024, 3072),
            torch.float32,
            "runtime",
            [sdpa_ex, transformer_engine_ex],
            [
                [sdpa_ex, transformer_engine_ex, nvfuserex, torchex, pythonex],
                [transformer_engine_ex, sdpa_ex, nvfuserex, torchex, pythonex],
            ],
            True,
        ),
    ],
)
@requiresCUDA
def test_autotuner(
    model: torch.nn.Module,
    tensor_shape: tuple,
    dtype: torch.dtype,
    autotune_type: str,
    executors: list,
    expected_executors: list[list],
    use_te: bool,
):
    def _run():
        model.to("cuda")
        x = torch.randn(tensor_shape, dtype=dtype, device="cuda")
        jitted_def = thunder.jit(model, executors=executors)
        jitted_auto = thunder.jit(
            model,
            autotune_type=autotune_type,
            executors=executors,
            autotune_enable_te=use_te,
        )
        y_def = jitted_def(x)
        y_auto = jitted_auto(x)

        te_used = aut_utils.is_te_used(thunder.last_traces(jitted_auto)[-1])
        got = thunder.executors_applied(jitted_auto)
        print("got", got)
        print("expected", expected_executors)
        assert any([t == got for t in expected_executors])
        # With TE enabled deviation ((y_def - y_auto).abs().max().item()) is between tensors are ~0.2
        # For the else branch: https://pytorch.org/docs/stable/testing.html
        torch.testing.assert_close(y_def, y_auto, atol=2 * 1e-1 if te_used else 1e-5, rtol=1e-1 if te_used else 1.3e-6)

    if dtype != torch.get_default_dtype():
        with torch.autocast(device_type="cuda"):
            _run()
    else:
        _run()


"""
The longest repeated block is:
    t2 = x @ y
    t3 = t0 + t0
    t4 = t1 * t1
"""


def _test_repetead_transformer_blocks_fn(x: torch.Tensor, y: torch.Tensor):
    t0 = x + x
    t1 = y * y
    t2 = x @ y
    t3 = t0 + t0
    t4 = t1 * t1
    t5 = t2 @ t2
    t6 = t3 + t3
    t7 = t4 * t4
    t8 = t6 - t7
    return t8, t5


def test_repetead_transformer_blocks():
    device = "cpu"

    a = torch.randn(2, 2, device=device)
    b = torch.randn(2, 2, device=device)

    jitted = thunder.jit(_test_repetead_transformer_blocks_fn, disable_dce=True)
    jitted(a, b)

    trace = thunder.last_traces(jitted)[-1]
    print(trace)
    blocks = aut_utils.repetead_trace_blocks(trace=trace)
    assert len(blocks) == 2
    assert blocks[0][1] - blocks[0][0] + 1 == 3


def test_reduce_common_trace_blocks():
    device = "cpu"

    a = torch.randn(2, 2, device=device)
    b = torch.randn(2, 2, device=device)

    jitted = thunder.jit(_test_repetead_transformer_blocks_fn, disable_dce=True)
    jitted(a, b)

    trace = thunder.last_traces(jitted)[-1]
    blocks = aut_utils.repetead_trace_blocks(trace=trace)
    reduced_trace = aut_utils.reduce_common_trace_blocks(
        trace=trace, common_blocks_in=blocks, skip_between_blocks=False
    )

    # We expect that t5, t6, t7 have been removed
    should_remove = set(["t5", "t6", "t7"])
    for b in reduced_trace.bound_symbols:
        if hasattr(b.output, "name"):
            assert b.output.name not in should_remove

@requiresCUDA
def test_save_configuration_cuda():
    class _LLaMAMLP(torch.nn.Module):
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

    with torch.device("cuda"):
        model = _LLaMAMLP(4, 4)
        jitted = thunder.jit(
            model,
            autotune_type="memory",
            model_name="llamamlp",
            autotune_save_configuration=True,
        )
        jitted_recovered = thunder.jit(
            model,
            autotune_type="runtime",
            autotune_restore_configuration="llamamlp_memory.json",
        )

        x = torch.randn(4, 4)
        a = jitted(x)
        b = jitted_recovered(x)

        torch.testing.assert_close(a, b)

        for bsym_a, bsym_b in zip(
            thunder.last_traces(jitted)[-1].bound_symbols, thunder.last_traces(jitted_recovered)[-1].bound_symbols
        ):
            assert bsym_a.sym.executor == bsym_b.sym.executor
