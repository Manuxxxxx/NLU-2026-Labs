import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GPT2_Mod


def train_loop(data, optimizer, criterion, model):
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.permute(0, 2, 1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array) / sum(number_of_tokens)).item())

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(
            data, desc="Evaluating:", unit="batch", total=len(data)
        ):
            output = model(input_ids)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


@dataclass
class ExperimentConfig:
    name: str
    d_model: int = 20
    n_heads: int = 1
    num_layers: int = 1
    ff_dim: int = 20
    lr: float = 0.01
    dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    emb_dropout: float = 0.0
    weight_tying: bool = False
    n_epochs: int = 10
    patience: int = 3
    warmup_epochs: int = 3
    save_best: bool = False


def _make_run_dir(name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_")
    base_dir = Path(__file__).resolve().parents[2]
    run_dir = base_dir / "runs" / f"1A_{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(
    cfg: ExperimentConfig,
    train_loader,
    dev_loader,
    test_loader,
    vocab_len: int,
    device: str,
    tokenizer,
):
    print(f"\n=== {cfg.name} ===")
    torch.manual_seed(42)
    model = GPT2_Mod(
        vocab_len,
        pos_emb_size=1024,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        proj_dropout=cfg.proj_dropout,
        emb_dropout=cfg.emb_dropout,
        weight_tying=cfg.weight_tying,
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    run_dir = _make_run_dir(cfg.name)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    checkpoint_path = run_dir / f"checkpoint_{cfg.name.replace(' ', '_')}.pt"

    best_ppl = math.inf
    best_model = copy.deepcopy(model).to("cpu")
    patience = cfg.patience
    warmup_epochs = cfg.warmup_epochs
    for epoch in tqdm(range(cfg.n_epochs), desc="Epochs"):
        loss = train_loop(train_loader, optimizer, criterion_train, model)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        if not math.isfinite(loss.item()) or not math.isfinite(ppl_dev):
            print("Non-finite loss/PPL detected, stopping early.")
            break
        writer.add_scalar("loss/train", loss.item(), epoch)
        writer.add_scalar("ppl/dev", ppl_dev, epoch)
        improved = ppl_dev < best_ppl
        if improved:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            if cfg.save_best:
                torch.save(best_model.state_dict(), checkpoint_path)
            if epoch + 1 > warmup_epochs:
                patience = cfg.patience
        else:
            if epoch + 1 > warmup_epochs:
                patience -= 1
        if epoch + 1 > warmup_epochs and patience <= 0:
            break

    best_model.to(device)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    if math.isfinite(final_ppl):
        writer.add_scalar("ppl/test", final_ppl)
    writer.close()
    print(f"Best dev PPL: {best_ppl:.2f} | Test PPL: {final_ppl:.2f}")
    return best_ppl, final_ppl
