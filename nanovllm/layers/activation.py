import torch
from torch import nn
from nanovllm.kernels import activation_triton as activation_ops
from nanovllm.kernels import fused_triton as fused_kernels
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out


class SiluAndMulQuant(SiluAndMul):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def __init__(self, act_sum: bool = False):
        super().__init__()
        if act_sum:
            self.forward = self.forward_with_act_sum
        else:
            self.forward = self.forward_wo_act_sum

    def forward_with_act_sum(
        self,
        x: torch.Tensor,
        quantized_mlp_act_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor,
    ) -> torch.Tensor:
        out = super().forward(x)
        fused_kernels.invoke_quant_fuse_sum(
            quantized_mlp_act_buffer, out, quantized_sum_buffer, quantized_scale_buffer
        )

    def forward_wo_act_sum(
        self,
        x: torch.Tensor,
        quantized_mlp_act_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor = None,
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface.
        out = super().forward(x)
        fused_kernels.invoke_quant(
            quantized_mlp_act_buffer, out, quantized_scale_buffer
        )
