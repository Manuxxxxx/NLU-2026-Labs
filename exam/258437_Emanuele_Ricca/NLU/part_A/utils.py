import json
import random
from collections import Counter

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def load_data(path):
	dataset = []
	with open(path) as f:
		dataset = json.loads(f.read())
	return dataset


def create_dev_split(train_raw, portion=0.10, seed=42):
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
		self.pad_token = pad_token
		self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
		self.slot2id = self.lab2id(slots, cls=cls)
		self.intent2id = self.lab2id(intents, pad=False, cls=False)
		self.id2word = {v: k for k, v in self.word2id.items()}
		self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != "cls"}
		self.id2intent = {v: k for k, v in self.intent2id.items()}

	def w2id(self, elements, cutoff=None, unk=True, cls=True):
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
		vocab = {}
		if pad:
			vocab["pad"] = self.pad_token
		for elem in elements:
			vocab[elem] = len(vocab)
		if cls:
			vocab["cls"] = self.pad_token
		return vocab


class IntentsAndSlots(data.Dataset):
	def __init__(self, dataset, lang, unk="unk", cls="cls", add_cls=True):
		self.utterances = []
		self.intents = []
		self.slots = []
		self.unk = unk
		self.cls = cls
		self.add_cls = add_cls

		for x in dataset:
			self.utterances.append(x["utterance"])
			self.slots.append(x["slots"])
			self.intents.append(x["intent"])

		self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
		self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
		self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

	def __len__(self):
		return len(self.utterances)

	def __getitem__(self, idx):
		utt = torch.Tensor(self.utt_ids[idx])
		slots = torch.Tensor(self.slot_ids[idx])
		intent = self.intent_ids[idx]
		sample = {"utterance": utt, "slots": slots, "intent": intent}
		return sample

	def mapping_lab(self, data, mapper):
		return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

	def mapping_seq(self, data, mapper):
		res = []
		for seq in data:
			tmp_seq = []
			for x in seq.split():
				if x in mapper:
					tmp_seq.append(mapper[x])
				else:
					tmp_seq.append(mapper[self.unk])
			if self.add_cls:
				tmp_seq.append(mapper[self.cls])
			res.append(tmp_seq)
		return res


def collate_fn(data, pad_token, device):
	def merge(sequences):
		lengths = [len(seq) for seq in sequences]
		max_len = 1 if max(lengths) == 0 else max(lengths)
		padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
		for i, seq in enumerate(sequences):
			end = lengths[i]
			padded_seqs[i, :end] = seq
		return padded_seqs, lengths

	data_by_key = {}
	for key in data[0].keys():
		data_by_key[key] = [d[key] for d in data]

	src_utt, _ = merge(data_by_key["utterance"])
	y_slots, y_lengths = merge(data_by_key["slots"])
	intent = torch.LongTensor(data_by_key["intent"])

	src_utt = src_utt.to(device)
	y_slots = y_slots.to(device)
	intent = intent.to(device)
	y_lengths = torch.LongTensor(y_lengths).to(device)

	new_item = {
		"utterances": src_utt,
		"intents": intent,
		"y_slots": y_slots,
		"slots_len": y_lengths,
	}
	return new_item


def build_loaders(train_dataset, dev_dataset, test_dataset, pad_token, device):
	train_loader = DataLoader(
		train_dataset, batch_size=128, collate_fn=lambda x: collate_fn(x, pad_token, device), shuffle=True
	)
	dev_loader = DataLoader(
		dev_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x, pad_token, device)
	)
	test_loader = DataLoader(
		test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x, pad_token, device)
	)
	return train_loader, dev_loader, test_loader
