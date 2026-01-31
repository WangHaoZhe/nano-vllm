import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,  # [M, N] Input
    w_ptr,  # [N] Weight (Gamma)
    y_ptr,  # [M, N] Output (int8)
    stride_x_row,
    stride_y_row,
    N,
    eps,
    use_quant,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptrs = x_ptr + pid * stride_x_row
    y_ptrs = y_ptr + pid * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptrs + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    if use_quant:
        y = x * rstd * w  # TODO: quant
    else:
        y = x * rstd * w

    y = y.to(tl.float16)

    tl.store(y_ptrs + cols, y.to(y_ptrs.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_kernel(
    x_ptr,  # [M, N] Input
    w_ptr,  # [N] Weight (Gamma)
    y_ptr,  # [M, N] Output (int8)
    scale_ptr,  # [M] Output (Quantization scale)
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptrs = x_ptr + pid * stride_x_row
    y_ptrs = y_ptr + pid * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptrs + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    y = (x - mean) * rstd * w

    # y = y.to(tl.float16)

    y_abs = tl.abs(y)
    amax = tl.max(y_abs, axis=0)
    amax = tl.maximum(amax, 1e-6)  # Avoid division by zero
    scale_val = 127.0 / amax

    y_q = tl.math.floor((y.to(tl.float32) * scale_val) + 0.5).to(tl.int8)  # Round

    tl.store(y_ptrs + cols, y_q.to(y_ptrs.dtype.element_ty), mask=mask)
    tl.store(scale_ptr + pid, (amax / 127.0).to(scale_ptr.dtype.element_ty))


@triton.jit
def _layernorm_fuse_sum_kernel(
    x_ptr,  # [M, N] Input
    w_ptr,  # [N] Weight (Gamma)
    y_ptr,  # [M, N] Output (int8)
    input_sum_ptr,  # [M] Output (Sum of float output)
    scale_ptr,  # [M] Output (Quantization scale)
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptrs = x_ptr + pid * stride_x_row
    y_ptrs = y_ptr + pid * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptrs + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    y = (x - mean) * rstd * w

    # y = y.to(tl.float16)
    y_sum = tl.sum(y, axis=0).to(tl.float16)

    y_abs = tl.abs(y)
    amax = tl.max(y_abs, axis=0)
    amax = tl.maximum(amax, 1e-6)  # Avoid division by zero
    scale_val = 127.0 / amax

    y_q = tl.math.floor((y * scale_val) + 0.5).to(tl.int8)  # Round

    tl.store(y_ptrs + cols, y_q.to(y_ptrs.dtype.element_ty), mask=mask)
    tl.store(scale_ptr + pid, (amax / 127.0).to(scale_ptr.dtype.element_ty))
    tl.store(input_sum_ptr + pid, y_sum.to(input_sum_ptr.dtype.element_ty))


def rms_norm(
    out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps=1e-6, use_quant=False
):
    """
    x: Input tensor [M, N]
    w: Weight tensor [N]
    out: Output tensor [M, N] (int8) - pre-allocated
    eps: Epsilon
    """
    M, N = x.shape
    assert w.shape[0] == N
    assert x.is_cuda and w.is_cuda
    assert out.shape == (M, N)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)
    _rms_norm_kernel[grid](
        x,
        w,
        out,
        x.stride(0),
        out.stride(0),
        N,
        eps=eps,
        use_quant=use_quant,
        BLOCK_N=BLOCK_N,
    )


def rms_norm_general(
    out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, eps=1e-6
):
    """
    x: Input tensor [M, N]
    w: Weight tensor [N]
    out: Output tensor [M, N] (int8) - pre-allocated
    scale: Output tensor [M] (float16/float32) - pre-allocated
    eps: Epsilon
    """
    M, N = x.shape
    assert w.shape[0] == N
    assert x.is_cuda and w.is_cuda
    assert out.shape == (M, N)
    assert scale.shape == (M,)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)
    _layernorm_kernel[grid](
        x,
        w,
        out,
        scale,
        x.stride(0),
        out.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )


def rms_norm_general_fuse_sum(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    input_sum: torch.Tensor,
    scale: torch.Tensor,
    eps=1e-6,
):
    """
    x: Input tensor [M, N]
    w: Weight tensor [N]
    out: Output tensor [M, N] (int8) - pre-allocated
    input_sum: Output tensor [M] (float16/float32) - pre-allocated
    scale: Output tensor [M] (float16/float32) - pre-allocated
    eps: Epsilon
    """
    M, N = x.shape
    assert w.shape[0] == N
    assert x.is_cuda and w.is_cuda
    assert out.shape == (M, N)
    assert input_sum.shape == (M,)
    assert scale.shape == (M,)

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)
    _layernorm_fuse_sum_kernel[grid](
        x,
        w,
        out,
        input_sum,
        scale,
        x.stride(0),
        out.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
