# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

from typing import Optional

import torch
# from xformers.ops import AttentionBias

class ActivationBuffer:
    """
    Pre-allocated Buffer for activation in the model.

    Args:
        model: The input model
        batched_seq_len: The batched sequence length. Sum of all the sequence lengths in the batch.
    """

    def __init__(self, model, batched_seq_len: int):
        self.model_class = model.__class__.__name__
        self.model_dtype = model.model.embed_tokens.weight.dtype
        self.device = model.model.embed_tokens.weight.device
        # self.chunk_prefill_size = model.model_config.chunk_prefill_size
        assert self.model_class in [
            "LlamaForCausalLM",
            "MixtralForCausalLM",
        ], f"model_class: {self.model_class} is currently not supported."
        assert (
            self.model_dtype == torch.float16
        ), f"model_dtype is expected to be fp16. Current: {self.model_dtype}."

        self.batched_seq_len = batched_seq_len
        assert (
            self.batched_seq_len > 0
        ), f"batched_seq_len is expected to be greater than 0 to allocate activation buffer. Current: {self.batched_seq_len}."

        self.q_size = model.q_size
        self.kv_size = model.kv_size
        self.intermediate_size = model.config.intermediate_size
        self.hidden_size = model.config.hidden_size

        self.allocate_activation_buffer()

    def allocate_activation_buffer(self):
        if self.model_class == "LlamaForCausalLM":
            self.__allocate_activation_buffer_llama()
        elif self.model_class == "MixtralForCausalLM":
            raise NotImplementedError("MixtralForCausalLM is not supported yet.")
        else:
            raise NotImplementedError(
                f"model_class: {self.model_class} is currently not supported."
            )

    def __allocate_activation_buffer_llama(self):
        # Allocate fp16 activation buffer.
        self.act_buffer = torch.empty(
            (
                self.batched_seq_len
                * max(self.q_size + 2 * self.kv_size, 2 * self.intermediate_size)
            ),
            device=self.device,
            dtype=torch.float16,
        )
        self.qkv_proj_act_buffer = self.act_buffer[
            : self.batched_seq_len * (self.q_size + 2 * self.kv_size)
        ].view(self.batched_seq_len, self.q_size + 2 * self.kv_size)
        self.out_down_proj_act_buffer = self.act_buffer[
            : self.batched_seq_len * self.hidden_size
        ].view(self.batched_seq_len, self.hidden_size)
        # self.gate_up_proj_act_buffer = self.act_buffer[
        #     : self.batched_seq_len * 2 * self.intermediate_size
        # ].view(self.batched_seq_len, 2 * self.intermediate_size)
        self.gate_up_proj_act_buffer = torch.empty(
            (min(4096, self.batched_seq_len), 2 * self.intermediate_size), device=self.device, dtype=torch.float16
        )

        # Allocate quantized activation buffer.
        self.quantized_act_buffer = torch.empty(
            (self.batched_seq_len * self.hidden_size),#(self.batched_seq_len * max(self.hidden_size, self.intermediate_size)),
            device=self.device,
            dtype=torch.int8,
        )
        self.quantized_hidden_states_buffer = self.quantized_act_buffer[
            : self.batched_seq_len * self.hidden_size
        ].view(self.batched_seq_len, self.hidden_size)
        # self.quantized_mlp_act_buffer = self.quantized_act_buffer[
        #     : self.batched_seq_len * self.intermediate_size
        # ].view(self.batched_seq_len, self.intermediate_size)
        self.quantized_mlp_act_buffer = torch.empty(
            (min(4096, self.batched_seq_len), self.intermediate_size), device=self.device, dtype=torch.int8
        )

        self.quantized_scale_buffer = torch.empty(
            (self.batched_seq_len), device=self.device, dtype=torch.float16
        )
        self.quantized_sum_buffer = torch.empty(
            (self.batched_seq_len), device=self.device, dtype=torch.float16
        )
