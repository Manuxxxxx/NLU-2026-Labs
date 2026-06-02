import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class LoRALinear(nn.Module):
	def __init__(self, in_features, out_features, rank, alpha):
		super().__init__()
		# Low-rank adapters with scaling.
		self.rank = rank
		self.alpha = alpha
		self.scaling = alpha / rank
		# Factorized update: in_features -> rank -> out_features.
		self.lora_A = nn.Linear(in_features, rank, bias=False)
		self.lora_B = nn.Linear(rank, out_features, bias=False)
		nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
		nn.init.zeros_(self.lora_B.weight)

	def forward(self, x):
		# Apply LoRA update on top of frozen weights.
		return self.lora_B(self.lora_A(x)) * self.scaling


class CustomGPT2Attention(GPT2Attention):
	def __init__(self, config, rank, alpha):
		super().__init__(config)
		# Inject LoRA adapters into Q/K/V projections.
		embed_dim = config.hidden_size
		self.lora_q = LoRALinear(embed_dim, embed_dim, rank, alpha)
		self.lora_k = LoRALinear(embed_dim, embed_dim, rank, alpha)
		self.lora_v = LoRALinear(embed_dim, embed_dim, rank, alpha)

	def forward(
		self,
		hidden_states: Optional[Tuple[torch.FloatTensor]],
		layer_past: Optional[Tuple[torch.Tensor]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = False,
		output_attentions: Optional[bool] = False,
	) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
		if encoder_hidden_states is not None:
			if not hasattr(self, "q_attn"):
				raise ValueError(
					"If class is used as cross attention, the weights `q_attn` have to be defined. "
					"Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
				)

			query = self.q_attn(hidden_states)
			key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
			attention_mask = encoder_attention_mask
			# LoRA updates for cross-attention path.
			key = key + self.lora_k(encoder_hidden_states)
			value = value + self.lora_v(encoder_hidden_states)
		else:
			query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
			# LoRA updates for self-attention path.
			query = query + self.lora_q(hidden_states)
			key = key + self.lora_k(hidden_states)
			value = value + self.lora_v(hidden_states)

		# Split heads for attention computation.
		query = self._split_heads(query, self.num_heads, self.head_dim)
		key = self._split_heads(key, self.num_heads, self.head_dim)
		value = self._split_heads(value, self.num_heads, self.head_dim)

		# Append cached KV states when using past.
		if layer_past is not None:
			past_key, past_value = layer_past
			key = torch.cat((past_key, key), dim=-2)
			value = torch.cat((past_value, value), dim=-2)

		if use_cache is True:
			present = (key, value)
		else:
			present = None

		if self.reorder_and_upcast_attn:
			attn_output, attn_weights = self._upcast_and_reordered_attn(
				query, key, value, attention_mask, head_mask
			)
		else:
			attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

		attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
		attn_output = self.c_proj(attn_output)
		attn_output = self.resid_dropout(attn_output)

		outputs = (attn_output, present)
		if output_attentions:
			outputs += (attn_weights,)

		return outputs


class GPT2_LoRA(GPT2LMHeadModel):
	def __init__(self, *model_args, rank, alpha, **model_kwargs):
		super().__init__(*model_args, **model_kwargs)
		# Swap attention modules with LoRA-augmented versions.
		for block in self.transformer.h:
			old_attn = block.attn
			new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
			# Reuse pretrained weights for the base attention.
			new_attn.load_state_dict(old_attn.state_dict(), strict=False)
			block.attn = new_attn
