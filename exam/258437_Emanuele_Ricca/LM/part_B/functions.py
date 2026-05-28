import copy
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GPT2_LoRA, LoRALinear


def param_stats(model):
	total = sum(param.numel() for param in model.parameters())
	trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
	print(f"total params: {total:,}")
	print(f"trainable params: {trainable:,}")
	print(f"frozen params: {total - trainable:,}")


def make_lora_trainable(model: nn.Module):
	for param in model.parameters():
		param.requires_grad = False
	for module in model.modules():
		if isinstance(module, LoRALinear):
			for param in module.parameters():
				param.requires_grad = True


def train_loop(data, optimizer, model, tokenizer):
	model.train()
	loss_array = []
	number_of_tokens = []

	pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

	for i, (input_ids, _, n_tokens) in enumerate(pbar):
		optimizer.zero_grad()
		labels = input_ids.clone().detach()
		labels[labels == tokenizer.pad_token_id] = -100
		output = model(input_ids, labels=labels)
		loss_array.append(output.loss.item() * n_tokens)
		number_of_tokens.append(n_tokens)
		output.loss.backward()
		optimizer.step()

		if i % 100 == 0:
			pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

	return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, model, tokenizer):
	model.eval()
	loss_array = []
	number_of_tokens = []
	with torch.no_grad():
		for input_ids, _, n_tokens in tqdm(data, desc="Evaluating:", unit="batch", total=len(data)):
			labels = input_ids.clone().detach()
			labels[labels == tokenizer.pad_token_id] = -100
			output = model(input_ids, labels=labels)
			loss_array.append(output.loss.item() * n_tokens)
			number_of_tokens.append(n_tokens)

	loss_to_return = sum(loss_array) / sum(number_of_tokens)
	ppl = math.exp(loss_to_return)
	return ppl, loss_to_return


@dataclass
class LoRAExperimentConfig:
	name: str
	rank: int = 8
	alpha: int = 16
	lr: float = 5e-4
	n_epochs: int = 50
	patience: int = 5
	warmup_epochs: int = 10


def _make_lora_run_dir(name: str) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	safe_name = name.replace(" ", "_")
	base_dir = Path(__file__).resolve().parents[2]
	run_dir = base_dir / "runs" / f"1B_{timestamp}_{safe_name}"
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def run_lora_experiment(cfg: LoRAExperimentConfig, train_loader, dev_loader, test_loader, device, tokenizer):
	print(f"\n=== {cfg.name} ===")
	torch.manual_seed(42)
	model = GPT2_LoRA.from_pretrained("openai-community/gpt2", alpha=cfg.alpha, rank=cfg.rank)
	model.to(device)
	make_lora_trainable(model)

	optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)

	run_dir = _make_lora_run_dir(cfg.name)
	writer = SummaryWriter(log_dir=str(run_dir / "tb"))
	checkpoint_path = run_dir / f"checkpoint_{cfg.name.replace(' ', '_')}.pt"

	best_ppl = math.inf
	best_model = copy.deepcopy(model).to("cpu")
	patience = cfg.patience
	warmup_epochs = cfg.warmup_epochs
	for epoch in tqdm(range(cfg.n_epochs), desc="Epochs"):
		loss = train_loop(train_loader, optimizer, model, tokenizer)
		ppl_dev, _ = eval_loop(dev_loader, model, tokenizer)
		if not math.isfinite(loss.item()) or not math.isfinite(ppl_dev):
			print("Non-finite loss/PPL detected, stopping early.")
			break
		writer.add_scalar("loss/train", loss.item(), epoch)
		writer.add_scalar("ppl/dev", ppl_dev, epoch)
		improved = ppl_dev < best_ppl
		if improved:
			best_ppl = ppl_dev
			best_model = copy.deepcopy(model).to("cpu")
			torch.save(best_model.state_dict(), checkpoint_path)
			if epoch + 1 > warmup_epochs:
				patience = cfg.patience
		else:
			if epoch + 1 > warmup_epochs:
				patience -= 1
		if epoch + 1 > warmup_epochs and patience <= 0:
			break

	best_model.to(device)
	final_ppl, _ = eval_loop(test_loader, best_model, tokenizer)
	if math.isfinite(final_ppl):
		writer.add_scalar("ppl/test", final_ppl)
	writer.close()
	print(f"Best dev PPL: {best_ppl:.2f} | Test PPL: {final_ppl:.2f}")
	return best_ppl, final_ppl
