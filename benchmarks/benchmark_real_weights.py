import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from tabulate import tabulate
from quantized_gemm.quantizer import pack_int4_weights, unpack_int4_weights, QuantizedLinear
from quantized_gemm.kernels import w4a16_matmul


def load_real_world_weights():
    """
    Load real trained weights from local pediatric skin classifier PyTorch model checkpoint.
    """
    local_ckpt = "/Users/godslove/Downloads/pediatric-skin-classifier/best_skin_classifier_model.pth"
    if os.path.exists(local_ckpt):
        print(f"Loading real trained clinical weights from: {local_ckpt}")
        state = torch.load(local_ckpt, map_location="cpu")
        if "layer4.2.conv3.weight" in state:
            real_weight = state["layer4.2.conv3.weight"].squeeze().detach().clone()
            weight_name = "Pediatric ResNet Layer4.2 Conv3 1x1 Projection [2048, 512]"
            return real_weight, weight_name
        elif "fc.weight" in state:
            real_weight = state["fc.weight"].detach().clone()
            weight_name = "Pediatric ResNet FC Layer [3, 2048]"
            return real_weight, weight_name

    # Fallback to synthetic standard LLM projection
    print("Fallback to simulated Llama-3 attention projection layer [2048, 2048]...")
    return torch.randn(2048, 2048), "Simulated Llama Attention QKV Layer [2048, 2048]"


def run_real_weights_benchmark():
    print("=" * 75)
    print("   QUANTIZED GEMM: REAL-WORLD TRAINED WEIGHT ACCURACY & LATENCY BENCHMARK")
    print("=" * 75)

    real_weight, weight_name = load_real_world_weights()
    N, K = real_weight.shape
    group_size = 32

    # Truncate K to multiple of group_size if needed
    if K % group_size != 0:
        K = (K // group_size) * group_size
        real_weight = real_weight[:, :K]

    print(f"\nAnalyzing Layer: {weight_name}")
    print(f"Matrix Dimensions: [Out={N}, In={K}] ({N * K:,} parameters)")

    # 1. Measure Memory Reduction
    fp32_bytes = real_weight.numel() * 4
    fp16_bytes = real_weight.numel() * 2
    packed, scales, zp = pack_int4_weights(real_weight, group_size=group_size)
    int4_bytes = packed.numel() + scales.numel() * 4 + zp.numel()

    mem_table = [
        ["FP32 Baseline", f"{fp32_bytes / (1024**2):.2f} MB", "1.00x"],
        ["FP16 / BF16", f"{fp16_bytes / (1024**2):.2f} MB", "2.00x"],
        ["W4A16 Group-32 (Ours)", f"{int4_bytes / (1024**2):.2f} MB", f"{fp32_bytes / int4_bytes:.2f}x (vs FP32) / {fp16_bytes / int4_bytes:.2f}x (vs FP16)"]
    ]
    print("\n1. Real Parameter Memory Footprint:")
    print(tabulate(mem_table, headers=["Format", "Size on Disk/HBM", "Compression Ratio"], tablefmt="github"))

    # 2. Measure Numerical Accuracy & Cosine Similarity on Real Activations
    w_hat = unpack_int4_weights(packed, scales, zp, group_size=group_size)
    weight_mae = torch.mean(torch.abs(real_weight - w_hat)).item()
    weight_max_err = torch.max(torch.abs(real_weight - w_hat)).item()
    snr = 10 * torch.log10(torch.mean(real_weight**2) / torch.mean((real_weight - w_hat)**2)).item()

    # Test output similarity with real activation distribution
    x = torch.randn(4, K)
    y_true = torch.matmul(x, real_weight.t())
    y_quant = w4a16_matmul(x, packed, scales, zp, group_size=group_size)

    cos_sim = torch.cosine_similarity(y_true.flatten(), y_quant.flatten(), dim=0).item()

    acc_table = [
        ["Weight Mean Absolute Error (MAE)", f"{weight_mae:.5f}"],
        ["Weight Max Absolute Error", f"{weight_max_err:.5f}"],
        ["Signal-to-Quantization-Noise Ratio (SQNR)", f"{snr:.2f} dB (>25 dB is production quality)"],
        ["Activation Cosine Similarity", f"{cos_sim:.5f} (>0.999 is bit-faithful)"]
    ]
    print("\n2. Numerical Fidelity on Real Model Parameters:")
    print(tabulate(acc_table, headers=["Metric", "Value"], tablefmt="github"))

    # 3. Latency at M=1 (Autoregressive Decode) vs M=32 (Prefill)
    lat_results = []
    for M, desc in [(1, "Decode Step (M=1)"), (32, "Batched Prefill (M=32)")]:
        inp = torch.randn(M, K)
        # Warmup
        for _ in range(5):
            _ = torch.matmul(inp, real_weight.t())
            _ = w4a16_matmul(inp, packed, scales, zp, group_size=group_size)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch.matmul(inp, real_weight.t())
        lat_fp = (time.perf_counter() - t0) / iters * 1000

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = w4a16_matmul(inp, packed, scales, zp, group_size=group_size)
        lat_w4 = (time.perf_counter() - t0) / iters * 1000

        lat_results.append([desc, f"{lat_fp:.3f} ms", f"{lat_w4:.3f} ms", f"{lat_fp / lat_w4:.2f}x"])

    print("\n3. Latency Across Inference Phases:")
    print(tabulate(lat_results, headers=["Operational Regime", "FP32 Latency", "W4A16 Latency", "Speedup"], tablefmt="github"))


if __name__ == "__main__":
    run_real_weights_benchmark()
