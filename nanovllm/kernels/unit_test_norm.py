USE_CUDA = 1
USE_TRITON = 1
DO_BENCHMARK = 1
NCU = 0

import torch
import triton
import sys
import os


# Add current directory to sys.path to allow importing omniserve modules
sys.path.append(os.getcwd())

# -------------------------------------------------------------------------
# 0. Prepare Parameters
# -------------------------------------------------------------------------
# Set device
device = torch.device("cuda")
print(f"Using device: {device}")

# Dimensions
M = 64  # Batch size / Sequence length
K = 4096  # Hidden size / Input features
N = 64  # Output features

print(f"Dimensions: M={M}, K={K}, N={N}")


# -------------------------------------------------------------------------
# 1. Prepare Deterministic Input (Hidden States)
# -------------------------------------------------------------------------
hidden_states = torch.arange(K, dtype=torch.float16, device=device).repeat(M, 1)
# Add some variation across rows to ensure M dimension is handled correctly
hidden_states += torch.arange(M, device=device).unsqueeze(1) * 0.1

print("\n--- Input Hidden States (FP16) ---")
print(hidden_states[0, :10])  # Print first 10 elements of first row

norm_weight = torch.ones(K, dtype=torch.float16, device=device)

# -------------------------------------------------------------------------
# 2. Prepare Functions
# -------------------------------------------------------------------------
if USE_TRITON:
    from layernorm_triton import (
        rms_norm_general_fuse_sum,
        rms_norm_general,
        rms_norm,
    )

    output_triton = torch.empty((M, K), dtype=torch.int8, device=device)
    # output_triton = torch.empty((M, K), dtype=torch.float16, device=device)
    quantized_scale_buffer_triton = torch.empty((M), dtype=torch.float16, device=device)
    quantized_sum_buffer_triton = torch.empty((M), dtype=torch.float16, device=device)

    def run_triton():
        rms_norm_general_fuse_sum(
            output_triton,
            hidden_states,
            norm_weight,
            quantized_sum_buffer_triton,
            quantized_scale_buffer_triton,
        )
        # rms_norm_general(
        #     output_triton,
        #     hidden_states,
        #     norm_weight,
        #     quantized_scale_buffer_triton,
        # )
        # rms_norm(
        #     output_triton,
        #     hidden_states,
        #     norm_weight,
        # )


if USE_CUDA:
    from omniserve_backend import layernorm_ops

    output_cuda = torch.empty((M, K), dtype=torch.int8, device=device)
    # output_cuda = torch.empty((M, K), dtype=torch.float16, device=device)
    quantized_scale_buffer_cuda = torch.empty((M), dtype=torch.float16, device=device)
    quantized_sum_buffer_cuda = torch.empty((M), dtype=torch.float16, device=device)

    def run_cuda():
        layernorm_ops.rms_norm_general_fuse_sum(
            output_cuda,
            hidden_states,
            norm_weight,
            quantized_sum_buffer_cuda,
            quantized_scale_buffer_cuda,
            epsilon=1e-6,
            use_per_token_quant=True,
        )
        # layernorm_ops.rms_norm_general(
        #     output_cuda,
        #     hidden_states,
        #     norm_weight,
        #     quantized_scale_buffer_cuda,
        #     epsilon=1e-6,
        #     use_per_token_quant=True,
        # )
        # layernorm_ops.rms_norm(
        #     output_cuda,
        #     hidden_states,
        #     norm_weight,
        #     epsilon=1e-6,
        #     use_quant=False,
        # )


compare_variables = {
    "output": [output_cuda, output_triton, 1],
    "quantized_scale_buffer": [
        quantized_scale_buffer_cuda,
        quantized_scale_buffer_triton,
        1e-4,
    ],
    "quantized_sum_buffer": [
        quantized_sum_buffer_cuda,
        quantized_sum_buffer_triton,
        1e-4,
    ],
}

# -------------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# -------------------------------------------------------------------------

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
for key in compare_variables:
    if USE_CUDA:
        print(f"CUDA {key} [0]: {compare_variables[key][0][0]}")
    if USE_TRITON:
        print(f"Triton {key} [0]: {compare_variables[key][1][0]}")


if USE_TRITON and USE_CUDA:
    print("\n--- Comparison Results ---")

    def compare(name, args):
        output_cuda, output_triton, tolerance = args
        diff = torch.abs(output_cuda - output_triton).to(float)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff <= tolerance:
            print(f"\n✅ TEST PASSED: Triton {name} matches CUDA output:")
            print(f"Max Difference: {max_diff:.6f}")
            print(f"Mean Difference: {mean_diff:.6f}")
        else:
            print(f"\n❌ TEST FAILED: Significant {name} difference detected:")
            print(f"Max Difference: {max_diff:.6f}")
            print(f"Mean Difference: {mean_diff:.6f}")
            mismatch_idx = torch.where(diff > tolerance)
            num_mismatches = len(mismatch_idx[0])
            print(f"\nFound {num_mismatches}/{diff.numel()} mismatches")
            print(f"{'Index':<10} | {'CUDA':<10} | {'Triton':<10} | {'Diff':<10}")
            for i in range(min(500, num_mismatches)):
                current_idx = tuple(idx[i].item() for idx in mismatch_idx)
                cuda_val = output_cuda[current_idx].item()
                triton_val = output_triton[current_idx].item()
                diff_val = diff[current_idx].item()
                idx_str = str(current_idx)
                print(
                    f"{idx_str:<10} | {cuda_val:<10.6f} | {triton_val:<10.6f} | {diff_val:<10.6f}"
                )

    for key in compare_variables:
        compare(key, compare_variables[key])

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
        print(f"CUDA Execution Time (Mid):   {ms_cuda[0]:.4f} ms")
        print(f"CUDA Execution Time (High 20%):   {ms_cuda[1]:.4f} ms")
        print(f"CUDA Execution Time (Low 20%):   {ms_cuda[2]:.4f} ms")
    if USE_TRITON:
        ms_triton = triton.testing.do_bench(run_triton, quantiles=[0.5, 0.2, 0.8])
        print(f"Triton Execution Time (Mid): {ms_triton[0]:.4f} ms")
        print(f"Triton Execution Time (High 20%): {ms_triton[1]:.4f} ms")
        print(f"Triton Execution Time (Low 20%): {ms_triton[2]:.4f} ms")

    if USE_TRITON and USE_CUDA:
        print(f"Speedup (CUDA / Triton): {ms_cuda[0] / ms_triton[0]:.2f}x")
