import pytest
import torch
import torch.nn as nn
from quantized_gemm.quantizer import pack_int4_weights, unpack_int4_weights, QuantizedLinear
from quantized_gemm.kernels import w4a16_matmul, w4a16_matmul_python


def test_pack_unpack_shapes():
    N, K = 128, 256
    group_size = 32
    w = torch.randn(N, K)

    packed, scales, zp = pack_int4_weights(w, group_size=group_size)

    assert packed.shape == (N, K // 2)
    assert packed.dtype == torch.uint8
    assert scales.shape == (N, K // group_size, 1)
    assert zp.shape == (N, K // group_size, 1)

    w_hat = unpack_int4_weights(packed, scales, zp, group_size=group_size)
    assert w_hat.shape == (N, K)


def test_quantization_reconstruction_fidelity():
    N, K = 256, 512
    group_size = 32
    torch.manual_seed(42)
    w = torch.randn(N, K)

    packed, scales, zp = pack_int4_weights(w, group_size=group_size)
    w_hat = unpack_int4_weights(packed, scales, zp, group_size=group_size)

    # Relative Frobenius norm error should be small for 4-bit (~5-10% max)
    rel_error = torch.norm(w - w_hat) / torch.norm(w)
    assert rel_error < 0.12, f"Quantization error too high: {rel_error.item():.4f}"

    # Pearson correlation between w and w_hat should be > 0.98
    vx = w - torch.mean(w)
    vy = w_hat - torch.mean(w_hat)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    assert corr > 0.98, f"Correlation too low: {corr.item():.4f}"


@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_group_sizes(group_size):
    N, K = 64, 256
    w = torch.randn(N, K)
    packed, scales, zp = pack_int4_weights(w, group_size=group_size)
    w_hat = unpack_int4_weights(packed, scales, zp, group_size=group_size)
    assert w_hat.shape == (N, K)


def test_quantized_linear_forward():
    torch.manual_seed(42)
    M, K, N = 8, 256, 128
    group_size = 32

    linear = nn.Linear(K, N, bias=True)
    q_linear = QuantizedLinear.from_float(linear, group_size=group_size)

    x = torch.randn(M, K)

    y_ref = linear(x)
    y_quant = q_linear(x)

    assert y_quant.shape == (M, N)
    rel_err = torch.norm(y_ref - y_quant) / torch.norm(y_ref)
    assert rel_err < 0.12, f"Output error too high: {rel_err.item():.4f}"


def test_batched_input_dimensions():
    B, S, K, N = 2, 16, 128, 64
    linear = nn.Linear(K, N, bias=False)
    q_linear = QuantizedLinear.from_float(linear, group_size=32)

    x = torch.randn(B, S, K)
    out = q_linear(x)
    assert out.shape == (B, S, N)


def test_cpp_python_parity_if_available():
    try:
        import quantized_gemm_cpu_ext
    except ImportError:
        pytest.skip("C++ extension not built yet.")

    M, K, N = 16, 256, 128
    group_size = 32
    torch.manual_seed(42)

    w = torch.randn(N, K)
    packed, scales, zp = pack_int4_weights(w, group_size=group_size)
    x = torch.randn(M, K)

    out_py = w4a16_matmul(x, packed, scales, zp, group_size=group_size, use_cpp=False)
    out_cpp = w4a16_matmul(x, packed, scales, zp, group_size=group_size, use_cpp=True)

    max_diff = torch.max(torch.abs(out_py - out_cpp)).item()
    assert max_diff < 1e-5, f"C++ and Python parity error: max diff = {max_diff}"

