import torch
from typing import Optional
from .quantizer import unpack_int4_weights

# Attempt to import compiled C++ extension if built
_CPP_AVAILABLE = False
try:
    import quantized_gemm_cpu_ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


def w4a16_matmul_python(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 32
) -> torch.Tensor:
    """
    Pure PyTorch vectorized reference implementation of W4A16 GEMM:
    De-quantizes packed weights on the fly and multiplies against input activations.
    
    Y = X @ W_hat^T + b
    where X is [..., K], W_hat is [N, K], Y is [..., N]
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])  # [M, K]

    # Dequantize weights
    w_dequant = unpack_int4_weights(
        packed_weights,
        scales,
        zero_points,
        group_size=group_size,
        dtype=x.dtype
    )  # [N, K]

    # GEMM: [M, K] @ [K, N] -> [M, N]
    out = torch.matmul(x_2d, w_dequant.t())

    if bias is not None:
        out = out + bias

    return out.view(*orig_shape[:-1], packed_weights.shape[0])


def w4a16_matmul(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 32,
    use_cpp: bool = True
) -> torch.Tensor:
    """
    Unified dispatcher for W4A16 Quantized GEMM.
    Automatically chooses high-performance C++ backend if compiled, else Python fallback.
    """
    if use_cpp and _CPP_AVAILABLE and x.is_cpu and x.dtype == torch.float32:
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        out = quantized_gemm_cpu_ext.w4a16_gemm_forward(
            x_2d,
            packed_weights.contiguous(),
            scales.contiguous().to(torch.float32),
            zero_points.contiguous(),
            group_size
        )
        if bias is not None:
            out += bias
        return out.view(*orig_shape[:-1], packed_weights.shape[0])
    
    return w4a16_matmul_python(x, packed_weights, scales, zero_points, bias, group_size)


quantized_gemm_forward = w4a16_matmul

