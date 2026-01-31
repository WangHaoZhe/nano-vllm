import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from typing import Dict, List, Optional

from nanovllm.layers.activation import SiluAndMul, SiluAndMulQuant
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

from nanovllm.layers.w4a8_linear import W4A8OF16LinearDynamicInputScale
from nanovllm.layers.layernorm import RMSNormGeneral
from nanovllm.kernels.fused_triton import invoke_quant_fuse_sum
from nanovllm.utils.activation_buffer import ActivationBuffer
from nanovllm.utils.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)

class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        max_position_embeddings: int = 8192,
        bias: bool = False,
        bias_o_proj: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.total_num_kv_heads = num_kv_heads
        num_kv_heads_replicas = max(1, tp_size // self.total_num_kv_heads)

        self.qkv_proj = W4A8OF16LinearDynamicInputScale(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads * num_kv_heads_replicas)
            * self.head_dim,
            bias=False,
            group_size=-1,
        )

        self.o_proj = W4A8OF16LinearDynamicInputScale(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            group_size=-1,
        )

        self.invoke_quant = self.invoke_quant_with_act_sum

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
    
    def invoke_quant_with_act_sum(self, activation_buffer, attn_output):
        invoke_quant_fuse_sum(
            activation_buffer.quantized_hidden_states_buffer,
            attn_output,
            activation_buffer.quantized_sum_buffer,
            activation_buffer.quantized_scale_buffer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        activation_buffer: ActivationBuffer
    ) -> torch.Tensor:
        # INT8 in, FP16 out for this module
        self.qkv_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.quantized_sum_buffer,
            activation_buffer.qkv_proj_act_buffer,
        )

        q, k, v = activation_buffer.qkv_proj_act_buffer.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = q.reshape(q.size(0), self.total_num_heads, self.head_dim)
        k = k.reshape(k.size(0), self.num_kv_heads, self.head_dim)
        v = v.reshape(v.size(0), self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(q.size(0), -1)

        # FP16 in, INT8 out
        self.invoke_quant(activation_buffer, attn_output)
        # INT8 in, FP16 out
        self.o_proj(
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.quantized_sum_buffer,
            activation_buffer.out_down_proj_act_buffer,
        )


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.gate_up_proj = W4A8OF16LinearDynamicInputScale(
            hidden_size, 2 * intermediate_size, bias=False, group_size=-1
        )
        self.down_proj = W4A8OF16LinearDynamicInputScale(
            intermediate_size, hidden_size, bias=False, group_size=-1
        )

        assert hidden_act == "silu"
        self.act_fn = SiluAndMulQuant(act_sum=True)

    def forward(self, activation_buffer: ActivationBuffer):
        # INT8 in, FP16 out
        seq_len = activation_buffer.batched_seq_len
        hidden_size = activation_buffer.hidden_size
        intermediate_size = activation_buffer.intermediate_size
        for start_idx in range(0, seq_len, 4096):
            end_idx = min(seq_len, start_idx + 4096)
            # INT8 in, FP16 out
            self.gate_up_proj(
                activation_buffer.quantized_hidden_states_buffer[start_idx: end_idx, :],
                activation_buffer.quantized_scale_buffer[start_idx: end_idx],
                activation_buffer.quantized_sum_buffer[start_idx: end_idx],
                activation_buffer.gate_up_proj_act_buffer[: end_idx - start_idx, :],
            )

            # FP16 in, INT8 out
            self.act_fn(
                activation_buffer.gate_up_proj_act_buffer[: end_idx - start_idx, :],
                activation_buffer.quantized_mlp_act_buffer[: end_idx - start_idx, :],
                activation_buffer.quantized_scale_buffer[: end_idx - start_idx],
                activation_buffer.quantized_sum_buffer[: end_idx - start_idx],
            )

            self.down_proj(
                activation_buffer.quantized_mlp_act_buffer[: end_idx - start_idx, :],
                activation_buffer.quantized_scale_buffer[: end_idx - start_idx],
                activation_buffer.quantized_sum_buffer[: end_idx - start_idx],
                activation_buffer.out_down_proj_act_buffer[start_idx: end_idx, :],
            )


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNormGeneral(
            config.hidden_size,
            act_sum=True,
            eps=config.rms_norm_eps,
            use_per_token_quant=True
        )
        self.post_attention_layernorm = RMSNormGeneral(
            config.hidden_size,
            act_sum=True,
            eps=config.rms_norm_eps,
            use_per_token_quant=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        activation_buffer: ActivationBuffer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FP16 in FP16 out
        # Self Attention
        residual = hidden_states
        # INT8 quantization
        self.input_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.quantized_sum_buffer,
        )
        # INT8 -> FP16
        self.self_attn(positions, activation_buffer)
        hidden_states = residual + activation_buffer.out_down_proj_act_buffer
        # Fully Connected
        residual = hidden_states
        # FP16 -> INT8
        self.post_attention_layernorm(
            hidden_states,
            activation_buffer.quantized_hidden_states_buffer,
            activation_buffer.quantized_scale_buffer,
            activation_buffer.quantized_sum_buffer,
        )
        # INT8 -> FP16
        self.mlp(activation_buffer)
        hidden_states = residual + activation_buffer.out_down_proj_act_buffer
        return hidden_states


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        # self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        activation_buffer: ActivationBuffer
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, activation_buffer)
        # hidden_states = self.norm(hidden_states)
        seq_len = hidden_states.size(0)
        for start_idx in range(0, seq_len, 4096):
                end_idx = min(seq_len, start_idx + 4096)
                hidden_states[start_idx: end_idx, :] = self.norm(hidden_states[start_idx: end_idx, :])
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "down_proj"]

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        activation_buffer = ActivationBuffer(self, input_ids.size(0))
        model_output = self.model(input_ids, positions, activation_buffer)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
    
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        col_weight_suffixes = ["weight"]
        row_weight_suffixes = ["weight"]

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # TODO fix the tp parallelism
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        tp_size = 1
        tp_rank = 0

        q_proj_shard_size = self.config.hidden_size // tp_size
        num_kv_heads_replicas = max(1, tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1, self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * num_kv_heads_per_gpu
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
                pass
                # continue
            if "norm" in name:
                continue

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[
                    shard_size * shard_id : shard_size * (shard_id + 1)
                ]
                if "s2_scales" in name or "s2_zeros" in name:
                    param_slice = param.data[:, offset : offset + shard_size]
                else:
                    param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                if "s2_scales" in name or "s2_zeros" in name:
                    shard_size = param.shape[1] // 2
                    param_slice = param.data[
                        :, stride_id * shard_size : (stride_id + 1) * shard_size
                    ]
                else:
                    shard_size = param.shape[0] // 2
                    param_slice = param.data[
                        shard_size * stride_id : shard_size * (stride_id + 1)
                    ]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )
