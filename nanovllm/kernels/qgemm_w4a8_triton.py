import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
    ]


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N', 'K'],
# )
@triton.jit
def gemm_w4a8_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    W_scales_ptr,
    A_scales_ptr,
    W_szs_ptr,
    A_ssums_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for W4A8 GEMM.
    A: [M, K] int8
    B: [N, K//2] int8 (packed 4-bit)
    C: [M, N] float16
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # Pre-compute N indices decomposition for B loading (CUDA packing layout)
    # Layout: N_blk, K_blk, N_sub3, K_sub2, K_sub1, N_sub2, K_sub3, N_sub1(packed)
    n_blk = offs_bn // 32
    rem_n = offs_bn % 32
    n_sub1 = (rem_n >> 4) & 1
    n_sub2 = (rem_n >> 3) & 1
    n_sub3 = rem_n & 7

    offs_k = tl.arange(0, BLOCK_K)

    # A pointers
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_current = k * BLOCK_K + offs_k

        # Load A
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (k_current[None, :] < K),
            other=0,
        )

        # Load B with CUDA-compatible unpacking
        # Decompose K indices
        k_blk = k_current // 32
        rem_k = k_current % 32
        k_sub1 = (rem_k >> 4) & 1
        k_sub2 = (rem_k >> 2) & 3
        k_sub3 = rem_k & 3

        # Compute packed offset in bytes
        # Strides derived from permutation: (0, 4, 3, 6, 1, 5, 2, 7)
        # S_nblk = 32 * stride_bn (stride_bn is K//2)
        # S_kblk = 512
        # S_n3 = 64
        # S_k2 = 16
        # S_k1 = 8
        # S_n2 = 4
        # S_k3 = 1

        b_offs = (
            n_blk[:, None] * (32 * stride_bn)
            + (k_blk[None, :] << 9)
            + (n_sub3[:, None] << 6)
            + (k_sub2[None, :] << 4)
            + (k_sub1[None, :] << 3)
            + (n_sub2[:, None] << 2)
            + k_sub3[None, :]
        )
        # b_offs = (
        #     n_blk[:, None] * (32 * stride_bn)
        #     + k_blk[None, :] * 512
        #     + n_sub3[:, None] * 64
        #     + k_sub2[None, :] * 16
        #     + k_sub1[None, :] * 8
        #     + n_sub2[:, None] * 4
        #     + k_sub3[None, :] * 1
        # )

        b_packed = tl.load(
            B_ptr + b_offs,
            mask=(offs_bn[:, None] < N) & (k_current[None, :] < K),
            other=0,
        )

        # Unpack
        b_lo = (b_packed & 0xF).to(tl.int8)
        b_hi = ((b_packed >> 4) & 0xF).to(tl.int8)
        b = tl.where(n_sub1[:, None] == 0, b_lo, b_hi)

        accumulator += tl.dot(a, tl.trans(b))

        a_ptrs += BLOCK_K * stride_ak

    w_scale = tl.load(W_scales_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    a_scale = tl.load(A_scales_ptr + offs_am, mask=offs_am < M, other=1.0)
    w_sz = tl.load(W_szs_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    a_ssum = tl.load(A_ssums_ptr + offs_am, mask=offs_am < M, other=0.0)

    # (Accumulator * wscale * ascale) - (w_sz * a_ssum)
    c = accumulator.to(tl.float32)
    c = c * w_scale[None, :] * a_scale[:, None] - w_sz[None, :] * a_ssum[:, None]

    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(
        c_ptrs, c.to(tl.float16), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N)
    )


def gemm_forward_triton(in_feats, kernel, wscales, ascales, w_szs, a_ssums, out_feats):
    """
    in_feats: [M, K] int8
    kernel: [N, K//2] int8 (packed int4)
    wscales: [N] float16
    ascales: [M] float16
    w_szs: [N] float16
    a_ssums: [M] float16
    out_feats: [M, N] float16
    """
    M, K = in_feats.shape
    N = out_feats.shape[-1]
    assert kernel.shape == (
        N,
        K // 2,
    ), f"kernel shape mismatch: need {(N, K // 2)}, got {kernel.shape}"

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_stages = 4
    num_warps = 2

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    gemm_w4a8_kernel[grid](
        in_feats,
        kernel,
        out_feats,
        wscales,
        ascales,
        w_szs,
        a_ssums,
        M,
        N,
        K,
        in_feats.stride(0),
        in_feats.stride(1),
        kernel.stride(0),
        kernel.stride(1),
        out_feats.stride(0),
        out_feats.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    # print("Best config:", gemm_w4a8_kernel.best_config)
