import torch
import torch.nn as nn
from typing import Tuple, Optional


def pack_int4_weights(
    weights: torch.Tensor,
    group_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize an FP32/FP16 weight matrix [N, K] into asymmetric 4-bit integers
    and pack them into uint8 [N, K // 2].

    Args:
        weights: Float tensor of shape [N, K] (output_dim, input_dim).
        group_size: Number of consecutive K-elements sharing a scale & zero-point.

    Returns:
        packed_weights: torch.uint8 tensor of shape [N, K // 2].
        scales: torch.float32 tensor of shape [N, K // group_size, 1].
        zero_points: torch.uint8 tensor of shape [N, K // group_size, 1].
    """
    assert weights.dim() == 2, f"Expected 2D matrix [N, K], got {weights.shape}"
    N, K = weights.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, f"group_size must be even for 4-bit packing"

    # Reshape into groups: [N, num_groups, group_size]
    num_groups = K // group_size
    w_grouped = weights.view(N, num_groups, group_size)

    # Compute min and max per group
    min_val = w_grouped.amin(dim=-1, keepdim=True)
    max_val = w_grouped.amax(dim=-1, keepdim=True)

    # Scale and zero-point: mapping [min_val, max_val] to [0, 15]
    scales = (max_val - min_val) / 15.0
    scales = torch.clamp(scales, min=1e-7)

    zero_points = torch.round(-min_val / scales).clamp(0, 15).to(torch.uint8)

    # Quantize to [0, 15]
    q_vals = torch.round((w_grouped - min_val) / scales).clamp(0, 15).to(torch.uint8)
    q_flat = q_vals.view(N, K)

    # Pack pairs of 4-bit values into 1 byte (uint8)
    # Even indices occupy low 4 bits (0-3), Odd indices occupy high 4 bits (4-7)
    q_low = q_flat[:, 0::2]
    q_high = q_flat[:, 1::2]
    packed_weights = (q_high << 4) | (q_low & 0x0F)

    return packed_weights, scales, zero_points


def unpack_int4_weights(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int = 32,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Unpack and dequantize packed uint8 weights [N, K // 2] back to floating point [N, K].

    Args:
        packed_weights: uint8 tensor of shape [N, K // 2].
        scales: float tensor of shape [N, K // group_size, 1].
        zero_points: uint8 tensor of shape [N, K // group_size, 1].
        group_size: Quantization group size.
        dtype: Output float dtype.

    Returns:
        Dequantized weight matrix of shape [N, K].
    """
    N, half_K = packed_weights.shape
    K = half_K * 2
    num_groups = K // group_size

    # Unpack low and high 4-bit values
    q_low = packed_weights & 0x0F
    q_high = (packed_weights >> 4) & 0x0F

    # Interleave low and high back to [N, K]
    unpacked_q = torch.empty((N, K), dtype=torch.uint8, device=packed_weights.device)
    unpacked_q[:, 0::2] = q_low
    unpacked_q[:, 1::2] = q_high

    # Reshape to groups and dequantize: (q - zp) * scale
    unpacked_grouped = unpacked_q.view(N, num_groups, group_size).to(dtype)
    zp_grouped = zero_points.to(dtype)
    scales_grouped = scales.to(dtype)

    dequantized = (unpacked_grouped - zp_grouped) * scales_grouped
    return dequantized.view(N, K)


class QuantizedLinear(nn.Module):
    """
    Memory-efficient Linear Layer backed by 4-bit packed weights.
    Reduces parameter storage by 4x (vs FP16) and enables on-the-fly dequantized GEMM.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 32,
        device=None,
        dtype=torch.float32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype

        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        num_groups = in_features // group_size

        # Packed weight parameter (uint8 buffer, not differentiable directly)
        self.register_buffer(
            "packed_weight",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "scales",
            torch.ones((out_features, num_groups, 1), dtype=dtype, device=device)
        )
        self.register_buffer(
            "zero_points",
            torch.zeros((out_features, num_groups, 1), dtype=torch.uint8, device=device)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, group_size: int = 32):
        """Construct QuantizedLinear from an existing standard nn.Linear layer."""
        with torch.no_grad():
            q_layer = cls(
                in_features=float_linear.in_features,
                out_features=float_linear.out_features,
                bias=float_linear.bias is not None,
                group_size=group_size,
                device=float_linear.weight.device,
                dtype=float_linear.weight.dtype
            )
            packed, scales, zp = pack_int4_weights(float_linear.weight, group_size=group_size)
            q_layer.packed_weight.copy_(packed)
            q_layer.scales.copy_(scales)
            q_layer.zero_points.copy_(zp)

            if float_linear.bias is not None:
                q_layer.bias.copy_(float_linear.bias)

        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing on-the-fly unpacking GEMM:
        Y = X @ W_dequantized^T + Bias
        """
        from .kernels import w4a16_matmul
        return w4a16_matmul(x, self.packed_weight, self.scales, self.zero_points, self.bias, self.group_size)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, "
            f"compression=4x (INT4 vs FP16)"
        )
