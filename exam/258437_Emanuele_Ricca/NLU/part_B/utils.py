import json
from collections import Counter
from typing import Dict, List

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

SLOT_IGNORE_INDEX = -100


def load_data(path):
	# Load ATIS JSON split.
	dataset = []
	with open(path) as f:
		dataset = json.loads(f.read())
	return dataset


def create_dev_split(train_raw, portion=0.10, seed=42):
	# Stratified split, keeping singletons in train.
	intents = [x["intent"] for x in train_raw]
	count_y = Counter(intents)

	labels = []
	inputs = []
	mini_train = []

	for id_y, y in enumerate(intents):
		if count_y[y] > 1:
			inputs.append(train_raw[id_y])
			labels.append(y)
		else:
			mini_train.append(train_raw[id_y])

	X_train, X_dev, y_train, y_dev = train_test_split(
		inputs,
		labels,
		test_size=portion,
		random_state=seed,
		shuffle=True,
		stratify=labels,
	)
	X_train.extend(mini_train)
	return X_train, X_dev, y_train, y_dev


class Lang:
	def __init__(self, words, intents, slots, cutoff=0, cls=True, pad_token=0):
		# Build vocabularies with optional CLS token.
		self.pad_token = pad_token
		self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
		self.slot2id = self.lab2id(slots, cls=cls)
		self.intent2id = self.lab2id(intents, pad=False, cls=False)
		self.id2word = {v: k for k, v in self.word2id.items()}
		self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != "cls"}
		self.id2intent = {v: k for k, v in self.intent2id.items()}

	def w2id(self, elements, cutoff=None, unk=True, cls=True):
		# Token vocabulary with PAD/UNK/CLS.
		vocab = {"pad": self.pad_token}
		if unk:
			vocab["unk"] = len(vocab)
		if cls:
			vocab["cls"] = len(vocab)
		count = Counter(elements)
		for k, v in count.items():
			if v > cutoff:
				vocab[k] = len(vocab)
		return vocab

	def lab2id(self, elements, pad=True, cls=True):
		# Label vocabulary for slots/intents.
		vocab = {}
		if pad:
			vocab["pad"] = self.pad_token
		for elem in elements:
			vocab[elem] = len(vocab)
		if cls:
			vocab["cls"] = self.pad_token
		return vocab


def align_labels_with_tokens(word_ids, slot_labels: List[str], slot2id: Dict[str, int]) -> List[int]:
	# Align word-level slot tags to subword tokenization.
	label_ids = []
	previous_word_id = None
	for word_id in word_ids:
		if word_id is None:
			label_ids.append(SLOT_IGNORE_INDEX)
		elif word_id != previous_word_id:
			label_ids.append(slot2id[slot_labels[word_id]])
		else:
			label_ids.append(SLOT_IGNORE_INDEX)
		previous_word_id = word_id
	return label_ids


class HFIntentSlotDataset(Dataset):
	def __init__(self, raw_data, lang, tokenizer, max_length: int = 128):
		# Store raw samples and tokenizer for on-the-fly encoding.
		self.raw_data = raw_data
		self.lang = lang
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.raw_data)

	def __getitem__(self, idx):
		example = self.raw_data[idx]
		words = example["utterance"].split()
		slots = example["slots"].split()
		intent_id = self.lang.intent2id[example["intent"]]

		# Tokenize and keep word_ids for slot alignment.
		encoding = self.tokenizer(
			words,
			is_split_into_words=True,
			truncation=True,
			max_length=self.max_length,
		)
		word_ids = encoding.word_ids()
		slot_labels = align_labels_with_tokens(word_ids, slots, self.lang.slot2id)

		item = {
			"input_ids": encoding["input_ids"],
			"attention_mask": encoding["attention_mask"],
			"token_type_ids": encoding.get("token_type_ids"),
			"slot_labels": slot_labels,
			"intent_id": intent_id,
			"words": words,
			"slots": slots,
			"word_ids": word_ids,
		}
		return item


def make_collate_fn(tokenizer):
	def collate(batch):
		# Pad inputs and slot labels to max sequence length.
		input_features = []
		for item in batch:
			features = {
				"input_ids": item["input_ids"],
				"attention_mask": item["attention_mask"],
			}
			if item["token_type_ids"] is not None:
				features["token_type_ids"] = item["token_type_ids"]
			input_features.append(features)

		padded = tokenizer.pad(input_features, padding=True, return_tensors="pt")
		max_len = padded["input_ids"].shape[1]
		slot_labels = torch.full((len(batch), max_len), SLOT_IGNORE_INDEX, dtype=torch.long)
		for i, item in enumerate(batch):
			labels = torch.tensor(item["slot_labels"], dtype=torch.long)
			slot_labels[i, : labels.shape[0]] = labels

		intents = torch.tensor([item["intent_id"] for item in batch], dtype=torch.long)

		return {
			"input_ids": padded["input_ids"],
			"attention_mask": padded["attention_mask"],
			"token_type_ids": padded.get("token_type_ids"),
			"slot_labels": slot_labels,
			"intent_ids": intents,
			"words": [item["words"] for item in batch],
			"slots": [item["slots"] for item in batch],
			"word_ids": [item["word_ids"] for item in batch],
		}

	return collate


def build_tokenizer(model_name: str):
	# GPT-2 needs prefix spaces for word alignment.
	if "gpt2" in model_name.lower():
		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def build_loaders(model_name: str, batch_size: int, max_length: int, train_raw, dev_raw, test_raw, lang):
	# Build HF datasets and loaders.
	tokenizer = build_tokenizer(model_name)
	train_dataset = HFIntentSlotDataset(train_raw, lang, tokenizer, max_length=max_length)
	dev_dataset = HFIntentSlotDataset(dev_raw, lang, tokenizer, max_length=max_length)
	test_dataset = HFIntentSlotDataset(test_raw, lang, tokenizer, max_length=max_length)

	collate_fn = make_collate_fn(tokenizer)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
	dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

	return tokenizer, train_loader, dev_loader, test_loader
