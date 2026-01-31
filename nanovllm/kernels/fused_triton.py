import torch
import triton
import triton.language as tl


@triton.jit
def _quant_kernel(
    x_ptr,  # Input: [M, N]
    y_ptr,  # Output: [M, N] (int8)
    scale_ptr,  # Output Scale: [M]
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_m
    row_y_ptr = y_ptr + pid * stride_y_m
    amax = 0.0

    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_x_n, mask=mask, other=0.0).to(tl.float32)

        abs_x = tl.abs(x)
        block_max = tl.max(abs_x, axis=0)
        amax = tl.maximum(amax, block_max)

    amax = tl.maximum(amax, 1e-6)
    scale_val = 127.0 / amax
    tl.store(scale_ptr + pid, (amax / 127.0).to(scale_ptr.dtype.element_ty))

    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_x_n, mask=mask, other=0.0).to(tl.float32)

        y_q = tl.math.floor((x * scale_val) + 0.5)

        tl.store(
            row_y_ptr + offs * stride_y_n, y_q.to(y_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _quant_fuse_sum_kernel(
    x_ptr,  # Input: [M, N]
    y_ptr,  # Output: [M, N] (int8)
    sum_ptr,  # Output Sum: [M]
    scale_ptr,  # Output Scale: [M]
    stride_x_m,
    stride_x_n,
    stride_y_m,
    stride_y_n,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_m
    row_y_ptr = y_ptr + pid * stride_y_m
    amax = 0.0
    total_sum = 0.0

    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_x_n, mask=mask, other=0.0).to(tl.float32)

        total_sum += tl.sum(x, axis=0)

        abs_x = tl.abs(x)
        block_max = tl.max(abs_x, axis=0)
        amax = tl.maximum(amax, block_max)
    tl.store(sum_ptr + pid, total_sum.to(sum_ptr.dtype.element_ty))

    amax = tl.maximum(amax, 1e-6)
    scale_val = 127.0 / amax
    tl.store(scale_ptr + pid, (amax / 127.0).to(scale_ptr.dtype.element_ty))

    for off in range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(row_x_ptr + offs * stride_x_n, mask=mask, other=0.0).to(tl.float32)

        y_q = tl.math.floor((x * scale_val) + 0.5)

        tl.store(
            row_y_ptr + offs * stride_y_n, y_q.to(y_ptr.dtype.element_ty), mask=mask
        )


def invoke_quant(out: torch.Tensor, input: torch.Tensor, scale: torch.Tensor):
    """
    Per-token dynamic quantization

    input: [M, N] input tensor (FP16/BF16/FP32)
    scale: [M] output tensor for scaling factors (FP16/BF16)
    output: [M, N] output tensor for quantized values (Int8)
    """
    M, N = input.shape
    assert out.shape == (M, N)
    assert scale.shape[0] == M

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _quant_kernel[grid](
        input,
        out,
        scale,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def invoke_quant_fuse_sum(
    out: torch.Tensor, input: torch.Tensor, input_sum: torch.Tensor, scale: torch.Tensor
):
    """
    Per-token dynamic quantization fused with sum reduction

    input: [M, N] input tensor (FP16/BF16/FP32)
    input_sum: [M] output tensor for sum values (FP16/BF16)
    scale: [M] output tensor for scaling factors (FP16/BF16)
    output: [M, N] output tensor for quantized values (Int8)
    """
    M, N = input.shape
    assert out.shape == (M, N)
    assert input_sum.shape[0] == M
    assert scale.shape[0] == M

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    num_tokens = input.numel() // N
    grid = (num_tokens,)

    _quant_fuse_sum_kernel[grid](
        input,
        out,
        input_sum,
        scale,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
