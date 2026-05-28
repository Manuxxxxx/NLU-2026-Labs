import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

from model import BertForIntentSlots, GPT2ForIntentSlots
from utils import SLOT_IGNORE_INDEX, build_loaders

from pathlib import Path as SysPath

sys.path.append(str(SysPath(__file__).resolve().parents[2]))
from conll import evaluate


@dataclass
class ExperimentConfigBase:
	name: str
	model_name: str
	lr: float = 2e-5
	batch_size: int = 16
	max_length: int = 128
	n_epochs: int = 5
	patience: int = 5
	warmup_epochs: int = 10
	dropout: float = 0.1
	weight_decay: float = 0.01
	seed: int = 42


@dataclass
class ExperimentConfigBert(ExperimentConfigBase):
	pass


@dataclass
class ExperimentConfigGPT(ExperimentConfigBase):
	pass


def train_loop_mtl(data, model, optimizer, criterion_slots, criterion_intents, device):
	model.train()
	loss_array = []

	for batch in data:
		optimizer.zero_grad()
		input_ids = batch["input_ids"].to(device)
		attention_mask = batch["attention_mask"].to(device)
		slot_labels = batch["slot_labels"].to(device)
		intent_ids = batch["intent_ids"].to(device)

		if isinstance(model, BertForIntentSlots):
			slot_logits, intent_logits = model(
				input_ids,
				attention_mask,
				token_type_ids=(batch["token_type_ids"].to(device) if batch["token_type_ids"] is not None else None),
			)
		else:
			slot_logits, intent_logits = model(input_ids, attention_mask)

		slot_logits = slot_logits.permute(0, 2, 1)
		loss_intent = criterion_intents(intent_logits, intent_ids)
		loss_slot = criterion_slots(slot_logits, slot_labels)
		loss = loss_intent + loss_slot
		loss_array.append(loss.item())

		loss.backward()
		optimizer.step()

	return loss_array


def eval_loop_mtl(data, model, criterion_slots, criterion_intents, lang, device):
	model.eval()
	loss_array = []

	ref_intents = []
	hyp_intents = []
	ref_slots = []
	hyp_slots = []

	with torch.no_grad():
		for batch in data:
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			slot_labels = batch["slot_labels"].to(device)
			intent_ids = batch["intent_ids"].to(device)

			if isinstance(model, BertForIntentSlots):
				slot_logits, intent_logits = model(
					input_ids,
					attention_mask,
					token_type_ids=(batch["token_type_ids"].to(device) if batch["token_type_ids"] is not None else None),
				)
			else:
				slot_logits, intent_logits = model(input_ids, attention_mask)

			slot_logits = slot_logits.permute(0, 2, 1)
			loss_intent = criterion_intents(intent_logits, intent_ids)
			loss_slot = criterion_slots(slot_logits, slot_labels)
			loss = loss_intent + loss_slot
			loss_array.append(loss.item())

			out_intents = [lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
			gt_intents = [lang.id2intent[x] for x in intent_ids.tolist()]
			ref_intents.extend(gt_intents)
			hyp_intents.extend(out_intents)

			pred_slots = torch.argmax(slot_logits, dim=1)
			for i in range(pred_slots.shape[0]):
				words = batch["words"][i]
				gold_slots = batch["slots"][i]
				word_ids = batch["word_ids"][i]
				pred_ids = pred_slots[i].tolist()

				pred_word_slots = []
				seen = set()
				for token_idx, word_id in enumerate(word_ids):
					if word_id is None or word_id in seen:
						continue
					seen.add(word_id)
					if token_idx < len(pred_ids):
						pred_label = lang.id2slot.get(pred_ids[token_idx], "O")
						pred_word_slots.append(pred_label)

				min_len = min(len(words), len(gold_slots), len(pred_word_slots))
				ref_slots.append([(words[j], gold_slots[j]) for j in range(min_len)])
				hyp_slots.append([(words[j], pred_word_slots[j]) for j in range(min_len)])

	try:
		results = evaluate(ref_slots, hyp_slots)
	except Exception as ex:
		print("Warning:", ex)
		results = {"total": {"f": 0}}

	report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
	return results, report_intent, loss_array


def _make_run_dir(run_name: str) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	safe_name = run_name.replace(" ", "_")
	base_dir = Path(__file__).resolve().parents[2]
	run_dir = base_dir / "runs" / f"2B_{timestamp}_{safe_name}"
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def run_experiment(cfg: ExperimentConfigBase, train_raw, dev_raw, test_raw, lang, device):
	print(f"\n=== {cfg.name} ===")
	torch.manual_seed(cfg.seed)

	tokenizer, train_loader, dev_loader, test_loader = build_loaders(
		cfg.model_name, cfg.batch_size, cfg.max_length, train_raw, dev_raw, test_raw, lang
	)

	n_intents = len(lang.intent2id)
	n_slots = len(lang.slot2id)

	if isinstance(cfg, ExperimentConfigBert):
		model = BertForIntentSlots(cfg.model_name, n_intents, n_slots, dropout=cfg.dropout).to(device)
	else:
		model = GPT2ForIntentSlots(cfg.model_name, n_intents, n_slots, dropout=cfg.dropout).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	criterion_slots = nn.CrossEntropyLoss(ignore_index=SLOT_IGNORE_INDEX)
	criterion_intents = nn.CrossEntropyLoss()

	run_dir = _make_run_dir(cfg.name)
	writer = SummaryWriter(log_dir=str(run_dir / "tb"))
	checkpoint_path = run_dir / f"checkpoint_{cfg.name.replace(' ', '_')}.pt"

	with open(run_dir / "run_config.json", "w") as f:
		json.dump(asdict(cfg), f, indent=2)

	best_f1 = 0.0
	patience = cfg.patience
	warmup_epochs = cfg.warmup_epochs

	for epoch in range(cfg.n_epochs):
		loss = train_loop_mtl(train_loader, model, optimizer, criterion_slots, criterion_intents, device)
		results_dev, intent_res, loss_dev = eval_loop_mtl(
			dev_loader, model, criterion_slots, criterion_intents, lang, device
		)

		train_loss = float(sum(loss) / max(len(loss), 1))
		dev_loss = float(sum(loss_dev) / max(len(loss_dev), 1))
		dev_f1 = results_dev["total"]["f"]
		dev_intent_acc = intent_res["accuracy"]

		writer.add_scalar("loss/train", train_loss, epoch)
		writer.add_scalar("loss/dev", dev_loss, epoch)
		writer.add_scalar("slot_f1/dev", dev_f1, epoch)
		writer.add_scalar("intent_acc/dev", dev_intent_acc, epoch)

		improved = dev_f1 > best_f1
		if improved:
			best_f1 = dev_f1
			torch.save(model.state_dict(), checkpoint_path)
			if epoch + 1 > warmup_epochs:
				patience = cfg.patience
		else:
			if epoch + 1 > warmup_epochs:
				patience -= 1

		print(
			f"Epoch {epoch}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | "
			f"Slot F1={dev_f1:.4f} | Intent Acc={dev_intent_acc:.4f}"
		)

		if epoch + 1 > warmup_epochs and patience <= 0:
			break

	model.load_state_dict(torch.load(checkpoint_path))
	results_test, intent_test, _ = eval_loop_mtl(
		test_loader, model, criterion_slots, criterion_intents, lang, device
	)
	test_f1 = results_test["total"]["f"]
	test_intent_acc = intent_test["accuracy"]

	writer.add_scalar("slot_f1/test", test_f1)
	writer.add_scalar("intent_acc/test", test_intent_acc)
	writer.add_hparams(
		hparam_dict=asdict(cfg),
		metric_dict={
			"best_dev_slot_f1": best_f1,
			"test_slot_f1": test_f1,
			"test_intent_acc": test_intent_acc,
		},
	)
	writer.close()

	print(
		f"Best dev Slot F1: {best_f1:.4f} | Test Slot F1: {test_f1:.4f} | "
		f"Test Intent Acc: {test_intent_acc:.4f}"
	)
	return cfg.name, best_f1, test_f1, test_intent_acc
