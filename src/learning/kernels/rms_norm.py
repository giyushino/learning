import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X,           # pointer to input, shape (M, N)
    Y,           # pointer to output, shape (M, N)
    W,           # pointer to weight, shape (N,)
    RSTD,        # pointer to saved reciprocal std, shape (M,) — needed for backward
    stride,      # number of elements between rows (usually N)
    N,           # number of columns
    eps,         # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    pass


@triton.jit
def _rms_norm_bwd_kernel(
    DX,          # pointer to output: grad w.r.t. x, shape (M, N)
    DW,          # pointer to output: partial grad w.r.t. w, shape (M, N) — caller sums over M
    DY,          # pointer to upstream gradient, shape (M, N)
    X,           # pointer to saved input, shape (M, N)
    W,           # pointer to weight, shape (N,)
    RSTD,        # pointer to saved reciprocal std, shape (M,)
    stride,      # number of elements between rows
    N,           # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    pass


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,      # (M, N)
        weight: torch.Tensor, # (N,)
        eps: float,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor,     # (M, N) upstream gradient
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        # returns (dx, dweight, None) — None for eps since it has no gradient
        pass


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return RMSNormFunction.apply(x, weight, eps)



