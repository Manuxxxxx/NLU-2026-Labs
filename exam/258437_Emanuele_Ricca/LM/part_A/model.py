import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention_Mod(nn.Module):
	def __init__(self, d_model, n_heads, attn_dropout=0.1, proj_dropout=0.1):
		super().__init__()
		# Split d_model across heads.
		assert d_model % n_heads == 0
		self.n_heads = n_heads
		self.h_dim = d_model // n_heads

		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)

		self.out_proj = nn.Linear(d_model, d_model)
		self.attn_dropout = nn.Dropout(attn_dropout)
		self.proj_dropout = nn.Dropout(proj_dropout)

	def forward(self, x, mask):
		B, L, d_model = x.size()
		# Project inputs to queries/keys/values.

		q = self.w_q(x)
		k = self.w_k(x)
		v = self.w_v(x)

		# (B, heads, L, head_dim)
		q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
		k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
		v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

		similarity = q @ k.transpose(-2, -1)
		# Scaled dot-product attention.
		similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim, device=x.device)))
		similarity = similarity.masked_fill(mask == 0, float("-inf"))

		# Softmax over key sequence length.
		attn = F.softmax(similarity, dim=-1)
		attn = self.attn_dropout(attn)
		y = attn @ v
		y = y.transpose(1, 2).contiguous().view(B, L, d_model)
		y = self.out_proj(y)
		y = self.proj_dropout(y)
		return y


class FeedForward_Mod(nn.Module):
	def __init__(self, d_model, hidden_dim, dropout=0.1):
		super().__init__()
		self.fc1 = nn.Linear(d_model, hidden_dim)
		self.act = nn.GELU()
		self.fc2 = nn.Linear(hidden_dim, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# Position-wise MLP.
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		x = self.dropout(x)
		return x


class TransformerBlock_Mod(nn.Module):
	def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, attn_dropout=0.1, proj_dropout=0.1):
		super().__init__()
		self.ln1 = nn.LayerNorm(d_model)
		self.attn = MultiHeadAttention_Mod(d_model, n_heads, attn_dropout, proj_dropout)
		self.ln2 = nn.LayerNorm(d_model)
		self.ff = FeedForward_Mod(d_model, ff_dim, dropout)

	def forward(self, x, mask):
		# Pre-norm Transformer block.
		x = x + self.attn(self.ln1(x), mask)
		x = x + self.ff(self.ln2(x))
		return x


class GPT2_Mod(nn.Module):
	def __init__(
		self,
		vocab_size,
		pos_emb_size=1024,
		d_model=768,
		n_heads=12,
		num_layers=12,
		ff_dim=3072,
		dropout=0.1,
		attn_dropout=0.1,
		proj_dropout=0.1,
		emb_dropout=0.1,
		weight_tying=False,
	):
		super().__init__()
		self.pos_emb_size = pos_emb_size
		self.token_embed = nn.Embedding(vocab_size, d_model)
		self.pos_embed = nn.Embedding(pos_emb_size, d_model)
		self.emb_dropout = nn.Dropout(emb_dropout)

		self.blocks = nn.ModuleList(
			[
				TransformerBlock_Mod(d_model, n_heads, ff_dim, dropout, attn_dropout, proj_dropout)
				for _ in range(num_layers)
			]
		)

		self.ln_f = nn.LayerNorm(d_model)
		self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

		# Optional weight tying between embedding and LM head.
		if weight_tying:
			self.lm_head.weight = self.token_embed.weight

		# Causal mask (lower triangular) for autoregressive attention.
		mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
		self.register_buffer("mask", mask)

	def forward(self, idx):
		B, L = idx.shape
		# Enforce max sequence length for positional embeddings.
		assert L <= self.pos_emb_size
		pos = torch.arange(L, device=idx.device)
		# Token + position embeddings.
		x = self.token_embed(idx) + self.pos_embed(pos)
		x = self.emb_dropout(x)
		mask = self.mask[:, :, :L, :L]
		for block in self.blocks:
			x = block(x, mask)
		x = self.ln_f(x)
		# Output logits for LM head.
		logits = self.lm_head(x)
		return logits
