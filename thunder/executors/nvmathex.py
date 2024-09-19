from importlib.metadata import version
from thunder.core.prims import PrimIDs
import logging
import thunder
import thunder.torch as ltorch
import torch

try:
    import nvmath
    HAS_NVMATH = True
except:
    pass
    HAS_NVMATH = False

logger = logging.getLogger("Thunder nvmath_ex")
logger.disabled = True

nvmath_ex = thunder.extend.OperatorExecutor("nvmath", version=version('nvmath-python'))
thunder.extend.register_executor(nvmath_ex)

_cache = {}
options = nvmath.linalg.advanced.MatmulOptions(logger=logger)

def _cache_key(a: torch.Tensor, b: torch.Tensor) -> str:
    def _get_shape_str(t: tuple):
        return '_'.join(str(num) for num in t)

    return f'{_get_shape_str(a.size())}-{_get_shape_str(b.size())}'

def _nvmath_linalg_advanced_matmul_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Check if these shapes have been cached
    k = _cache_key(a, b)
    if k in _cache:
        algo = _cache[k]
        with nvmath.linalg.advanced.Matmul(a, b, options=options) as mm:
            # Provide the optimized algorithms directly to plan.
            mm.plan(algorithms=algo)
            # Execute the multiplication
            return mm.execute()

    # Compute a new shape and cache the result
    with nvmath.linalg.advanced.Matmul(a, b, options=options) as mm:
        preferences = nvmath.linalg.advanced.MatmulPlanPreferences(limit=25)
        mm.plan(preferences=preferences)
        mm.autotune(iterations=10)
        # Execute the multiplication
        result = mm.execute()
        _cache[k] = mm.algorithms
        return result

def _nvmath_linalg_advanced_matmul_checker(*args, **kwargs) -> bool:
    return HAS_NVMATH

nvmath_linalg_advanced_matmul = nvmath_ex.register_operator(
    "nvmath_linalg_advanced_matmul",
    like=ltorch.matmul,
    fn=_nvmath_linalg_advanced_matmul_impl,
)
nvmath_ex.register_implementation(
    ltorch.matmul, nvmath_linalg_advanced_matmul, checker=_nvmath_linalg_advanced_matmul_checker
)

nvmath_ex.register_implementation(
    PrimIDs.MATMUL, nvmath_linalg_advanced_matmul, checker=_nvmath_linalg_advanced_matmul_checker
)
