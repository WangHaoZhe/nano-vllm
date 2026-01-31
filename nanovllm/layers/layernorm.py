import torch
import torch.nn as nn
# from omniserve_backend import layernorm_ops
from nanovllm.kernels import layernorm_triton as layernorm_ops


class RMSNorm(nn.Module):

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, use_quant: bool = False
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            torch.empty_like(x, dtype=torch.int8)
            if self.use_quant
            else torch.empty_like(x)
        )
        layernorm_ops.rms_norm(
            out, x, self.weight.data, self.variance_epsilon, self.use_quant
        )
        return out


class RMSNormGeneral(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        act_sum: bool = False,  # for per-channel weight quant, we need to pre-compute the sum of activation
        eps: float = 1e-6,
        use_per_token_quant: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_per_token_quant = use_per_token_quant
        if act_sum:
            self.forward = self.forward_with_act_sum
        else:
            self.forward = self.forward_wo_act_sum

    def forward_wo_act_sum(
        self,
        x: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor = None,
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface
        layernorm_ops.rms_norm_general(
            quantized_hidden_states_buffer,
            x,
            self.weight.data,
            quantized_scale_buffer,
            self.variance_epsilon,
        )

    def forward_with_act_sum(
        self,
        x: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor,
    ) -> torch.Tensor:
        layernorm_ops.rms_norm_general_fuse_sum(
            quantized_hidden_states_buffer,
            x,
            self.weight.data,
            quantized_sum_buffer,
            quantized_scale_buffer,
            self.variance_epsilon,
        )
