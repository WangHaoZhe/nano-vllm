USE_CUDA = 1
USE_TRITON = 1
DO_BENCHMARK = 1
NCU = 0

import torch
import triton
import sys
import os
import time


# Add current directory to sys.path to allow importing omniserve modules
sys.path.append(os.getcwd())

# -------------------------------------------------------------------------
# 0. Prepare Parameters
# -------------------------------------------------------------------------
# Set device
device = torch.device("cuda")
print(f"Using device: {device}")

# Dimensions
batch_size = 64
d = 28672  # Output dimension
input_dim = 2 * d  # Input dimension

print(f"Dimensions: batch_size={batch_size}, d={d}")

# -------------------------------------------------------------------------
# 1. Prepare Deterministic Input (Hidden States)
# -------------------------------------------------------------------------
torch.manual_seed(0)
input_tensor = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)

print("\n--- Input Tensor (FP32) ---")
print(input_tensor[0, :10])  # Print first 10 elements of first row

# -------------------------------------------------------------------------
# 2. Prepare Functions
# -------------------------------------------------------------------------
if USE_TRITON:
    from activation_triton import silu_and_mul

    output_triton = torch.empty(batch_size, d, device=device, dtype=torch.float32)

    def run_triton():
        silu_and_mul(output_triton, input_tensor)


if USE_CUDA:
    from omniserve_backend import activation_ops

    output_cuda = torch.empty(batch_size, d, device=device, dtype=torch.float32)

    def run_cuda():
        activation_ops.silu_and_mul(output_cuda, input_tensor)


# -------------------------------------------------------------------------
# 3. Run CUDA
# -------------------------------------------------------------------------
if USE_CUDA:
    run_cuda()
    torch.cuda.synchronize()

# -------------------------------------------------------------------------
# 4. Run Triton
# -------------------------------------------------------------------------
if USE_TRITON:
    run_triton()
    torch.cuda.synchronize()

if NCU:
    exit()

# -------------------------------------------------------------------------
# 5. Compare Results
# -------------------------------------------------------------------------
print("\n--- Results ---")
if USE_TRITON:
    print("Triton Output [0]:", output_triton[0])
if USE_CUDA:
    print("CUDA Output [0]:", output_cuda[0])

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
# 6. Performance Benchmark
# -------------------------------------------------------------------------
if DO_BENCHMARK:
    print("\n--- Performance Benchmark ---")

    # Warmup
    for _ in range(10):
        if USE_CUDA:
            run_cuda()
        if USE_TRITON:
            run_triton()
    torch.cuda.synchronize()

    # Benchmark
    if USE_CUDA:
        ms_cuda = triton.testing.do_bench(run_cuda, quantiles=[0.5, 0.2, 0.8])
    if USE_TRITON:
        ms_triton = triton.testing.do_bench(run_triton, quantiles=[0.5, 0.2, 0.8])

    if USE_CUDA:
        print(f"CUDA Execution Time (Mid):   {ms_cuda[0]:.4f} ms")
        print(f"CUDA Execution Time (High 20%):   {ms_cuda[1]:.4f} ms")
        print(f"CUDA Execution Time (Low 20%):   {ms_cuda[2]:.4f} ms")
    if USE_TRITON:
        print(f"Triton Execution Time (Mid): {ms_triton[0]:.4f} ms")
        print(f"Triton Execution Time (High 20%): {ms_triton[1]:.4f} ms")
        print(f"Triton Execution Time (Low 20%): {ms_triton[2]:.4f} ms")

    if USE_TRITON and USE_CUDA:
        print(f"Speedup (CUDA / Triton): {ms_cuda[0] / ms_triton[0]:.2f}x")
