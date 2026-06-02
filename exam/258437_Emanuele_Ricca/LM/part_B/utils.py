from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def read_file(path, eos_token="<eos>"):
	# Read PTB lines and append EOS token.
	output = []
	with open(path, "r") as f:
		for line in f.readlines():
			output.append(line.strip() + " " + eos_token)
	return output


class PennTreeBank(data.Dataset):
	def __init__(self, corpus):
		# Keep raw sentences for tokenization later.
		self.sents = [sent for sent in corpus]

	def __len__(self):
		return len(self.sents)

	def __getitem__(self, idx):
		return self.sents[idx]


def build_tokenizer(model_name="openai-community/gpt2"):
	# GPT-2 uses EOS as padding for causal LM.
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def collate_fn(batch, tokenizer, device):
	tokenized = tokenizer(batch, padding=True, return_tensors="pt")
	# Shift inputs/labels by one for next-token prediction.
	input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
	labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

	# Count non-pad tokens for loss normalization.
	n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

	return input_ids, labels, n_tokens


def build_loaders(
	train_dataset,
	dev_dataset,
	test_dataset,
	tokenizer,
	device,
	train_batch_size=8,
	eval_batch_size=16,
):
	# Train loader shuffles; dev/test keep order.
	train_loader = DataLoader(
		train_dataset,
		batch_size=train_batch_size,
		collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
		shuffle=True,
	)
	dev_loader = DataLoader(
		dev_dataset,
		batch_size=eval_batch_size,
		collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=eval_batch_size,
		collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device),
	)
	return train_loader, dev_loader, test_loader
