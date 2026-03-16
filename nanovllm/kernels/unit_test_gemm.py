USE_CUDA = 0
USE_TRITON = 1
USE_TORCH = 1
DO_BENCHMARK = 1
NCU = 0

import torch
import sys
import os
import time

# Add current directory to sys.path to allow importing omniserve modules
sys.path.append(os.getcwd())

import triton
from layernorm_triton import rms_norm_general_fuse_sum
from qgemm_w4a8_triton import gemm_forward_triton

if USE_CUDA:
    import omniserve_backend.qgemm_w4a8_per_chn

# Set device
device = torch.device("cuda")
print(f"Using device: {device}")

# Dimensions
M = 1024  # Batch size / Sequence length
K = 4096  # Hidden size / Input features
N = 4096  # Output features

print(f"Dimensions: M={M}, K={K}, N={N}")

# -------------------------------------------------------------------------
# 1. Prepare Deterministic Input (Hidden States)
# -------------------------------------------------------------------------
# Pattern: 0, 1, 2, ..., K-1 repeated M times
# This creates a predictable input pattern
hidden_states = torch.arange(K, dtype=torch.float16, device=device).repeat(M, 1)
# Add some variation across rows to ensure M dimension is handled correctly
hidden_states += torch.arange(M, device=device).unsqueeze(1) * 0.1

print("\n--- Input Hidden States (FP16) ---")
print(hidden_states[0, :10])  # Print first 10 elements of first row

# -------------------------------------------------------------------------
# 2. Prepare Buffers
# -------------------------------------------------------------------------
quantized_hidden_states_buffer = torch.empty((M, K), dtype=torch.int8, device=device)
quantized_scale_buffer = torch.empty((M), dtype=torch.float16, device=device)
quantized_sum_buffer = torch.empty((M), dtype=torch.float16, device=device)

# -------------------------------------------------------------------------
# 3. Run RMSNorm (Pre-processing)
# -------------------------------------------------------------------------
# We use a dummy weight of all 1s for RMSNorm to keep it simple
norm_weight = torch.ones(K, dtype=torch.float16, device=device)

rms_norm_general_fuse_sum(
    quantized_hidden_states_buffer,
    hidden_states,
    norm_weight,
    quantized_sum_buffer,
    quantized_scale_buffer,
)
torch.cuda.synchronize()

print("\n--- RMSNorm Output (Quantized Input for GEMM) ---")
print(
    "Quantized Hidden States (Int8) [First 10]:", quantized_hidden_states_buffer[0, :10]
)
print("Quantized Scales (FP16) [First 5]:", quantized_scale_buffer[:5])

# -------------------------------------------------------------------------
# 4. Prepare Deterministic Weights for GEMM
# -------------------------------------------------------------------------
# qweight: [N, K/2] int8.
# We want to test unpacking logic with varying values.
# We use values in [1, 7] to ensure packed byte fits in signed int8 [-128, 127].
# (7 << 4) | 7 = 119 < 127.
rows = torch.arange(N, device=device).unsqueeze(1)
cols = torch.arange(K // 2, device=device).unsqueeze(0)

# Generate varying values for low and high 4-bits based on position
val_lo = ((rows + cols) % 7) + 1
val_hi = ((rows + cols + 3) % 7) + 1  # Add offset to make hi different from lo

packed_val = (val_hi << 4) | val_lo
qweight = packed_val.to(torch.int8)
"""
# qweight: [N, K/2] int8.
# We want to test unpacking logic.
# Value 17 (0x11) means both low 4-bit and high 4-bit are 1.
# Value 34 (0x22) means both are 2.
# Let's use a pattern: Row 0 is all 1s, Row 1 is all 2s, etc.
qweight = torch.zeros((N, K // 2), dtype=torch.int8, device=device)
for i in range(N):
    val = (i % 7) + 1 # 1, 2, ..., 7
    packed_val = (val << 4) | val
    qweight[i, :] = packed_val
"""
# Weight Scales: [N] float16
# Set to 0.5 to test scaling logic
s1_scales = torch.full((N,), 0.5, dtype=torch.float16, device=device)

# Weight Zeros (Scaled): [N] float16
# Set to 0.1 to test zero point logic
s1_szeros = torch.full((N,), 0.1, dtype=torch.float16, device=device)

print("\n--- GEMM Weights ---")
print("QWeight (Int8 packed) [Row 0, First 8]:", qweight[0, :8])
print("QWeight (Int8 packed) [Row 1, First 8]:", qweight[1, :8])

# -------------------------------------------------------------------------
# 5. Run CUDA GEMM (Reference)
# -------------------------------------------------------------------------
torch.save(quantized_hidden_states_buffer, "./save/quantized_hidden_states_buffer.pt")
torch.save(qweight, "./save/qweight.pt")
torch.save(s1_scales, "./save/s1_scales.pt")
torch.save(quantized_scale_buffer, "./save/quantized_scale_buffer.pt")
torch.save(s1_szeros, "./save/s1_szeros.pt")
torch.save(quantized_sum_buffer, "./save/quantized_sum_buffer.pt")
# '''
quantized_hidden_states_buffer = torch.load("./save/quantized_hidden_states_buffer.pt")
qweight = torch.load("./save/qweight.pt")
s1_scales = torch.load("./save/s1_scales.pt")
quantized_scale_buffer = torch.load("./save/quantized_scale_buffer.pt")
s1_szeros = torch.load("./save/s1_szeros.pt")
quantized_sum_buffer = torch.load("./save/quantized_sum_buffer.pt")

# Standard Torch Implementation
if USE_TORCH:
    # Dequantize weights for torch.matmul comparison
    # qweight is [N, K/2] packed int8 (4-bit). unpack to [N, K]
    # We need to reverse the packing logic:
    # packed_val = (val_hi << 4) | val_lo
    unpack_lo = (qweight.view(torch.uint8) & 0x0F).to(torch.float16)
    unpack_hi = ((qweight.view(torch.uint8) >> 4) & 0x0F).to(torch.float16)
    
    # Interleave generic weight unpacking if K dim is packed? 
    # Usually in these kernels, it's often packed as [w0, w1] or similar.
    # Based on: val_lo is cols, val_hi is cols. 
    # If packed as [w0, w1], then we stack them.
    # Let's assume layout is [e0, e1, e2 ... eK] where e_2i is lo, e_2i+1 is hi, or similar.
    # Actually looking at generation:
    # rows = N, cols = K // 2
    # packed_val = (val_hi << 4) | val_lo
    # It seems one element of qweight contains two weight values.
    # Let's construct the full weight matrix [N, K].
    
    # NOTE: The exact unpacking order depends on the kernel's expectation.
    # Assuming standard packing: (w[i, 2j+1] << 4) | w[i, 2j]
    
    w_unpacked = torch.empty((N, K), dtype=torch.float16, device=device)
    w_unpacked[:, 0::2] = unpack_lo
    w_unpacked[:, 1::2] = unpack_hi
    
    # Apply scales and zeros
    # Weights = (qweight - zero) * scale
    # s1_scales: [N], s1_szeros: [N]
    
    scales_expanded = s1_scales.unsqueeze(1) # [N, 1]
    zeros_expanded = s1_szeros.unsqueeze(1)  # [N, 1]
    
    weights_fp16 = (w_unpacked - zeros_expanded) * scales_expanded
    
    # Transpose for matmul: [M, K] @ [K, N] -> [M, N]
    # weights_fp16 is [N, K], so we transpose it to [K, N]
    weights_fp16_t = weights_fp16.t()
    
    output_torch = torch.matmul(hidden_states, weights_fp16_t)
    # Note: The custom kernel might be doing (Input - Zero) * Scale * Weight + ...
    # But usually W4A8 kernels are: (Input_int8 - Input_zero) * Input_scale * Weight_dequant
    # The custom kernel has: quantized_hidden_states_buffer, quantized_scale_buffer, quantized_sum_buffer
    
    # If the kernel implements: Output = (Sum(X_q * W_q) - Sum(X_q)*W_zero)*Scaling...
    # It's safer to compare against the "ideal" logic which is simply FP16 Matmul.
    torch.cuda.synchronize()

if USE_CUDA:
    output_cuda = torch.empty((M, N), dtype=torch.float16, device=device)
    omniserve_backend.qgemm_w4a8_per_chn.gemm_forward_cuda(
        quantized_hidden_states_buffer,
        qweight,
        s1_scales,
        quantized_scale_buffer,
        s1_szeros,
        quantized_sum_buffer,
        output_cuda,
    )
    torch.cuda.synchronize()

# -------------------------------------------------------------------------
# 6. Run Triton GEMM (Target)
# -------------------------------------------------------------------------
if USE_TRITON:
    output_triton = torch.empty((M, N), dtype=torch.float16, device=device)
    gemm_forward_triton(
        quantized_hidden_states_buffer,
        qweight,
        s1_scales,
        quantized_scale_buffer,
        s1_szeros,
        quantized_sum_buffer,
        output_triton,
    )
    torch.cuda.synchronize()

if NCU:
    exit()

# -------------------------------------------------------------------------
# 7. Compare Results
# -------------------------------------------------------------------------
if USE_TRITON:
    print("Triton Output [0]:", output_triton[0])
if USE_CUDA:
    print("CUDA Output [0]:", output_cuda[0])
if USE_TORCH:
    print("Torch Output [0]:", output_torch[0])

if USE_TORCH and (USE_TRITON or USE_CUDA):
    print("\n--- Comparison with Torch (FP16) ---")
    # Note: W4A8 vs FP16 will have precision loss.
    # We just want to see if they are in the same ballpark.
    target = output_triton if USE_TRITON else output_cuda
    diff_torch = torch.abs(output_torch - target)
    print(f"Max Diff (Torch vs Target): {diff_torch.max().item():.6f}")
    print(f"Mean Diff (Torch vs Target): {diff_torch.mean().item():.6f}")

if USE_TRITON and USE_CUDA:
    print("\n--- Comparison Results ---")
    diff = torch.abs(output_cuda - output_triton)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nMax Difference: {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")

    if max_diff < 1e-2:  # Allow small tolerance for FP16 precision differences
        print("\n✅ TEST PASSED: Triton output matches CUDA output.")
    else:
        print("\n❌ TEST FAILED: Significant difference detected.")
        # Debugging info
        mismatch_idx = torch.where(diff > 1e-2)
        if len(mismatch_idx[0]) > 0:
            r, c = mismatch_idx[0][0], mismatch_idx[1][0]
            print(f"First mismatch at [{r}, {c}]:")
            print(f"  CUDA: {output_cuda[r, c]}")
            print(f"  Triton: {output_triton[r, c]}")
            print(f"  Diff: {diff[r, c]}")


# -------------------------------------------------------------------------
# 8. Performance Benchmark
# -------------------------------------------------------------------------
def run_cuda():
    omniserve_backend.qgemm_w4a8_per_chn.gemm_forward_cuda(
        quantized_hidden_states_buffer,
        qweight,
        s1_scales,
        quantized_scale_buffer,
        s1_szeros,
        quantized_sum_buffer,
        output_cuda,
    )


def run_triton():
    gemm_forward_triton(
        quantized_hidden_states_buffer,
        qweight,
        s1_scales,
        quantized_scale_buffer,
        s1_szeros,
        quantized_sum_buffer,
        output_triton,
    )

def run_torch():
    torch.matmul(hidden_states, weights_fp16_t)

if DO_BENCHMARK:
    print("\n--- Performance Benchmark ---")

    # Warmup
    for _ in range(10):
        if USE_CUDA:
            run_cuda()
        if USE_TRITON:
            run_triton()
        if USE_TORCH:
            run_torch()
    torch.cuda.synchronize()

    # Benchmark
    if USE_CUDA:
        ms_cuda = triton.testing.do_bench(run_cuda, quantiles=[0.5, 0.2, 0.8])
    if USE_TRITON:
        ms_triton = triton.testing.do_bench(run_triton, quantiles=[0.5, 0.2, 0.8])
    if USE_TORCH:
        ms_torch = triton.testing.do_bench(run_torch, quantiles=[0.5, 0.2, 0.8])

    if USE_CUDA:
        print(f"CUDA Execution Time (Mid):   {ms_cuda[0]:.4f} ms")
        print(f"CUDA Execution Time (High 20%):   {ms_cuda[1]:.4f} ms")
        print(f"CUDA Execution Time (Low 20%):   {ms_cuda[2]:.4f} ms")
    if USE_TRITON:
        print(f"Triton Execution Time (Mid): {ms_triton[0]:.4f} ms")
        print(f"Triton Execution Time (High 20%): {ms_triton[1]:.4f} ms")
        print(f"Triton Execution Time (Low 20%): {ms_triton[2]:.4f} ms")
    if USE_TORCH:
        print(f"Torch Execution Time (Mid):  {ms_torch[0]:.4f} ms")

    if USE_TRITON and USE_CUDA:
        print(f"Speedup (CUDA / Triton): {ms_cuda[0] / ms_triton[0]:.2f}x")
    if USE_TRITON and USE_TORCH:
        print(f"Speedup (Torch / Triton): {ms_torch[0] / ms_triton[0]:.2f}x")
