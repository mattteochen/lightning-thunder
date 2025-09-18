from typing import List, Set, Tuple
import thunder
import torch
from thunder.core.proxies import TensorProxy, Proxy
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx


def _numel(shape):
    n = 1
    for d in shape:
        try:
            n *= int(d)
        except Exception:
            return 0
    return n


def _elem_size_bytes(dtype):
    try:
        # dtype from Thunder -> torch dtype
        torch_dtype = thunder.dtypes.to_torch_dtype(dtype)
    except Exception:
        torch_dtype = None
    try:
        if torch_dtype is None:
            raise RuntimeError
        return torch.tensor([], dtype=torch_dtype).element_size()
    except Exception:
        # Fallback heuristics
        s = str(dtype).lower()
        if any(k in s for k in ("f16", "half", "bf16")):
            return 2
        if any(k in s for k in ("f64", "double")):
            return 8
        if any(k in s for k in ("i64", "long")):
            return 8
        if any(k in s for k in ("i16", "short")):
            return 2
        if any(k in s for k in ("i8", "u8", "bool")):
            return 1
        return 4  # default f32/i32


def _tensor_nbytes(p: TensorProxy) -> int:
    return _numel(getattr(p, "shape", ())) * _elem_size_bytes(getattr(p, "dtype", None))


class MemoryUsageStep:
    def __init__(self, step: int, mem_after: int, live_set: Set[str], step_bsym: BoundSymbol):
        self.step = step
        self.mem_after = mem_after
        self.live_set = live_set
        self.step_bsym = step_bsym

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"MemoryUsageStep(step={self.step}, mem_after={self.mem_after / (1024**2):.4f} MB, live_tensors_after_step={len(self.live_set)}, step_id='{self.step_bsym.sym.id}')"


class TraceMemoryUsage:
    def __init__(
        self,
        trace: TraceCtx,
        mem_after: List[MemoryUsageStep],
        peak_bytes: int,
        peak_step: int,
        live_ranges: List[Tuple[int, int, int]],
    ):
        self.trace = trace
        self.mem_after = mem_after
        self.peak_bytes = peak_bytes
        self.peak_step = peak_step
        self.live_ranges = live_ranges

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        from pprint import pformat

        return f"\n[Static Liveness]\npeak_bytes={self.peak_bytes / (1024**2):.4f} MB at step {self.peak_step}\n{pformat(self.mem_after)}"


def estimate_memory_usage(trace: TraceCtx) -> TraceMemoryUsage:
    bsyms = list(trace.bound_symbols)

    # Creation site (first producing bsym idx) and last-use site (max consuming idx)
    create_idx = {}
    last_use_idx = {}
    proxy_bytes = {}

    def _mark_use(p: Proxy, idx: int):
        if not isinstance(p, TensorProxy):
            return
        last_use_idx[id(p)] = max(last_use_idx.get(id(p), -1), idx)
        proxy_bytes.setdefault(id(p), _tensor_nbytes(p))

    for i, bs in enumerate(bsyms):
        # Record outputs (creation)
        for p in getattr(bs, "flat_proxy_outs", ()):
            if isinstance(p, TensorProxy):
                create_idx.setdefault(id(p), i)
                proxy_bytes.setdefault(id(p), _tensor_nbytes(p))
        # Record inputs (uses)
        for p in getattr(bs, "flat_proxy_args", ()):
            _mark_use(p, i)

    # Treat inputs as created at -1 and last used at their max use
    for key, lu in list(last_use_idx.items()):
        if key not in create_idx:
            create_idx[key] = -1

    # Build step-wise memory after each bsym: free tensors whose last_use == i, then allocate outputs at i
    mem_after: List[MemoryUsageStep] = []
    live_set = set()
    current_bytes = 0

    # Pre-populate any tensors considered "live" before first op (inputs)
    for key, ci in create_idx.items():
        if ci == -1:
            current_bytes += proxy_bytes.get(key, 0)
            live_set.add(key)

    # For each step, free after use, then allocate outputs
    last_use_to_keys = {}
    for key, lu in last_use_idx.items():
        last_use_to_keys.setdefault(lu, []).append(key)
    create_to_keys = {}
    for key, ci in create_idx.items():
        create_to_keys.setdefault(ci, []).append(key)

    for i, bsym in enumerate(bsyms):
        # Free tensors whose last use is this step
        for key in last_use_to_keys.get(i, []):
            if key in live_set:
                current_bytes -= proxy_bytes.get(key, 0)
                live_set.discard(key)
        # Allocate tensors created at this step
        for key in create_to_keys.get(i, []):
            current_bytes += proxy_bytes.get(key, 0)
            live_set.add(key)
        mem_after.append(MemoryUsageStep(i, max(current_bytes, 0), set(live_set), bsym.from_bsym()))

    peak_bytes = 0
    peak_step = -1
    for i, b in enumerate(mem_after):
        if b.mem_after > peak_bytes:
            peak_bytes, peak_step = b.mem_after, i

    # Build per-proxy ranges (start,end]
    live_ranges = []
    for key, start in create_idx.items():
        end = last_use_idx.get(key, start)
        live_ranges.append((start, end, proxy_bytes.get(key, 0)))

    return TraceMemoryUsage(trace, mem_after, peak_bytes, peak_step)
