from thunder import TensorProxy
from thunder.core.prims import PrimIDs
import nvmath
import thunder
import thunder.torch as ltorch
import torch

nvmath_ex = thunder.extend.OperatorExecutor('nvmath', version='0.1.0')
thunder.extend.register_executor(nvmath_ex)

def _nvmath_linalg_advanced_matmul_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nvmath.linalg.advanced.matmul(a, b)

def _nvmath_linalg_advanced_matmul_checker(a: TensorProxy, b: TensorProxy) -> bool:
    if len(a.shape) < 2 or len(b.shape) < 2:
        return False
    if a.shape[-1] != b.shape[-2]:
        return False
    if a.device != b.device:
        return False
    if a.dtype != b.dtype:
        return False
    # Handle distribuited
    return True

nvmath_linalg_advanced_matmul = nvmath_ex.register_operator(
    "nvmath_linalg_advanced_matmul",
    like=ltorch.matmul,
    fn=_nvmath_linalg_advanced_matmul_impl,
)
nvmath_ex.register_implementation(
    ltorch.matmul,
    nvmath_linalg_advanced_matmul,
    checker=_nvmath_linalg_advanced_matmul_checker
)

nvmath_ex.register_implementation(
    PrimIDs.MATMUL,
    nvmath_linalg_advanced_matmul,
    checker=_nvmath_linalg_advanced_matmul_checker
)
