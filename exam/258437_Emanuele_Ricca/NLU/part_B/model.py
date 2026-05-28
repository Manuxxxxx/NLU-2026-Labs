import torch
import torch.nn as nn
from transformers import AutoModel


class BertForIntentSlots(nn.Module):
	def __init__(self, model_name: str, n_intents: int, n_slots: int, dropout: float = 0.1):
		super().__init__()
		self.backbone = AutoModel.from_pretrained(model_name)
		hidden_size = self.backbone.config.hidden_size
		self.dropout = nn.Dropout(dropout)
		self.intent_head = nn.Linear(hidden_size, n_intents)
		self.slot_head = nn.Linear(hidden_size, n_slots)

	def forward(self, input_ids, attention_mask, token_type_ids=None):
		outputs = self.backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		hidden_states = outputs.last_hidden_state
		cls_state = hidden_states[:, 0]
		intent_logits = self.intent_head(self.dropout(cls_state))
		slot_logits = self.slot_head(self.dropout(hidden_states))
		return slot_logits, intent_logits


class GPT2ForIntentSlots(nn.Module):
	def __init__(self, model_name: str, n_intents: int, n_slots: int, dropout: float = 0.1):
		super().__init__()
		self.backbone = AutoModel.from_pretrained(model_name)
		hidden_size = self.backbone.config.hidden_size
		self.dropout = nn.Dropout(dropout)
		self.intent_head = nn.Linear(hidden_size, n_intents)
		self.slot_head = nn.Linear(hidden_size, n_slots)

	def forward(self, input_ids, attention_mask):
		outputs = self.backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
		)
		hidden_states = outputs.last_hidden_state
		slot_logits = self.slot_head(self.dropout(hidden_states))

		last_idx = attention_mask.sum(dim=1) - 1
		batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
		last_states = hidden_states[batch_idx, last_idx]
		intent_logits = self.intent_head(self.dropout(last_states))
		return slot_logits, intent_logits
