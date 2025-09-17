import torch.nn as nn
import thunder
import torch

model = nn.Sequential(nn.Linear(2048, 4096, dtype=torch.float16), nn.ReLU(), nn.Linear(4096, 64, dtype=torch.float16))

from thunder.recipes import BaseRecipe
from thunder.core.proxies import TensorProxy, Proxy


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


def static_liveness_memory(trace, include_inputs: bool = False):
    """Compute a coarse static liveness peak on a Thunder computation trace.

    Returns: dict with per-step bytes, peak info, and per-proxy live ranges.
    """
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
        for p in getattr(bs, "flat_proxy_args", () ):
            _mark_use(p, i)

    # Optionally include input tensors (those without a creation inside computation)
    if include_inputs:
        # Treat inputs as created at -1 and last used at their max use
        for key, lu in list(last_use_idx.items()):
            if key not in create_idx:
                create_idx[key] = -1

    # Build step-wise memory after each bsym: free tensors whose last_use == i, then allocate outputs at i
    steps = len(bsyms)
    mem_after = [0] * steps
    live_set = set()
    current_bytes = 0

    # Pre-populate any tensors considered "live" before first op (include_inputs)
    if include_inputs:
        for key, ci in create_idx.items():
            if ci == -1:
                current_bytes += proxy_bytes.get(key, 0)
                live_set.add(key)

    # For each step, free after use, then allocate outputs
    # Build reverse maps for efficiency
    last_use_to_keys = {}
    for key, lu in last_use_idx.items():
        last_use_to_keys.setdefault(lu, []).append(key)
    create_to_keys = {}
    for key, ci in create_idx.items():
        create_to_keys.setdefault(ci, []).append(key)

    for i in range(steps):
        # Free tensors whose last use is this step
        for key in last_use_to_keys.get(i, []):
            if key in live_set:
                current_bytes -= proxy_bytes.get(key, 0)
                live_set.discard(key)
        # Allocate tensors created at this step
        for key in create_to_keys.get(i, []):
            current_bytes += proxy_bytes.get(key, 0)
            live_set.add(key)
        mem_after[i] = max(current_bytes, 0)

    peak_bytes = 0
    peak_step = -1
    for i, b in enumerate(mem_after):
        if b > peak_bytes:
            peak_bytes, peak_step = b, i

    # Build per-proxy ranges (start,end]
    live_ranges = []
    for key, start in create_idx.items():
        end = last_use_idx.get(key, start)
        live_ranges.append((start, end, proxy_bytes.get(key, 0)))

    return {
        "mem_after": mem_after,
        "peak_bytes": peak_bytes,
        "peak_step": peak_step,
        "live_ranges": live_ranges,
    }

# Build a recipe and set the executor order by name (priority = list order)
r = BaseRecipe(interpreter="thunder.jit", fuser="torch.compile")  # or "torch.compile"
r.executor_names = ["torch"]  # any registered executors

thunder_model = thunder.compile(model, recipe=r)

x = torch.randn(64, 2048, dtype=torch.float16)
y = thunder_model(x)

traces = thunder.last_traces(thunder_model)
print(traces[-1])

last_trace = thunder.last_traces(thunder_model)[-1]

print('Bound Symbols:')
for b in last_trace.bound_symbols:
    print(b)

# --- Static liveness (mock) ---
report = static_liveness_memory(last_trace, include_inputs=False)
print("\n[Static Liveness]")
print(f"peak_bytes={report['peak_bytes']:,} at step {report['peak_step']}")
print("mem_after per step:", report["mem_after"]) 
