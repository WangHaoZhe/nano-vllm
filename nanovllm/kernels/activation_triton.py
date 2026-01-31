import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    M,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_input_ptr = input_ptr + pid * stride_xm
    row_output_ptr = output_ptr + pid * stride_ym

    for off_n in range(0, d, BLOCK_SIZE):
        cols = off_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < d

        x_ptrs = row_input_ptr + cols * stride_xn
        y_ptrs = row_input_ptr + (cols + d) * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)

        output = silu(x) * y

        output_ptrs = row_output_ptr + cols * stride_yn
        tl.store(output_ptrs, output.to(output_ptr.type.element_ty), mask=mask)


def silu_and_mul(output: torch.Tensor, input: torch.Tensor):
    """
    Triton kernel for silu_and_mul.

    Input: [..., 2 * d]
    Output: [..., d]
    """
    assert input.shape[-1] % 2 == 0
    d = input.shape[-1] // 2
    num_tokens = input.numel() // (2 * d)

    input_2d = input.view(num_tokens, 2 * d)
    output = output.view(num_tokens, d)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(d))

    grid = (num_tokens,)

    _silu_and_mul_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        output.stride(1),
        num_tokens,
        d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
