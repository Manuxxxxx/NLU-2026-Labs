import copy
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GPT2_Mod

import sys
from pathlib import Path as SysPath

sys.path.append(str(SysPath(__file__).resolve().parents[2]))
from conll import evaluate


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    # One full pass over the training set.
    model.train()
    loss_array = []

    for batch in tqdm(data, desc="Training:", unit="batch", total=len(data)):
        optimizer.zero_grad()

        slots, intent = model(batch["utterances"], batch["slots_len"])
        slots = slots.permute(0, 2, 1)

        loss_intent = criterion_intents(intent, batch["intents"])
        loss_slot = criterion_slots(slots, batch["y_slots"])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    # Evaluation without gradients.
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating:", unit="batch", total=len(data)):
            slots, intents = model(batch["utterances"], batch["slots_len"])
            slots = slots.permute(0, 2, 1)
            loss_intent = criterion_intents(intents, batch["intents"])
            loss_slot = criterion_slots(slots, batch["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                # Exclude the final CLS token from slot decoding.
                length = batch["slots_len"].tolist()[id_seq] - 1

                utt_ids = batch["utterances"][id_seq][:length].tolist()
                gt_ids = batch["y_slots"][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


def init_weights(mat):
    # Match reference init for linear layers.
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
    lr: float = 1e-3
    dropout: float = 0.1  # using 0.1, as the reference, not sure why though
    out_dropout: float = 0.0
    n_epochs: int = 100
    patience: int = 5
    warmup_epochs: int = 10
    test_after_epoch: int = 5
    save_best: bool = True


def _make_run_dir(name: str) -> Path:
    # Unique run folder per experiment.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_")
    base_dir = Path(__file__).resolve().parents[2]
    run_dir = base_dir / "runs" / f"2A_{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(
    cfg: ExperimentConfig,
    train_loader,
    dev_loader,
    test_loader,
    vocab_len,
    slots_len,
    n_intents,
    device,
    lang,
):
    print(f"\n=== {cfg.name} ===")
    # Deterministic initialization for comparability.
    torch.manual_seed(42)
    model = GPT2_Mod(
        vocab_len,
        slots_len,
        n_intents,
        pos_emb_size=1024,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        out_dropout=cfg.out_dropout,
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=0)
    criterion_intents = nn.CrossEntropyLoss()

    run_dir = _make_run_dir(cfg.name)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    checkpoint_path = run_dir / f"checkpoint_{cfg.name.replace(' ', '_')}.pt"

    best_f1 = 0.0
    best_model = copy.deepcopy(model).to("cpu")
    patience = cfg.patience
    warmup_epochs = cfg.warmup_epochs
    for epoch in tqdm(range(cfg.n_epochs), desc="Epochs"):
        loss = train_loop(
            train_loader, optimizer, criterion_slots, criterion_intents, model
        )
        results_dev, intent_res, _ = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )

        if cfg.test_after_epoch > 0 and epoch % cfg.test_after_epoch == 0:
            results_test_epoch, intent_test_epoch, _ = eval_loop(
                test_loader, criterion_slots, criterion_intents, model, lang
            )
            writer.add_scalar("slot_f1/test", results_test_epoch["total"]["f"], epoch)
            writer.add_scalar("intent_acc/test", intent_test_epoch["accuracy"], epoch)
            print(
                f"Epoch {epoch}: Test Slot F1={results_test_epoch['total']['f']:.4f} | "
                f"Test Intent Acc={intent_test_epoch['accuracy']:.4f}"
            )

        f1 = results_dev["total"]["f"]
        intent_acc = intent_res["accuracy"]
        writer.add_scalar("loss/train", float(sum(loss) / max(len(loss), 1)), epoch)
        writer.add_scalar("slot_f1/dev", f1, epoch)
        writer.add_scalar("intent_acc/dev", intent_acc, epoch)
        # Track best dev F1.
        improved = f1 > best_f1
        if improved:
            best_f1 = f1
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
        print(f"Epoch {epoch}: Slot F1={f1:.4f} | Intent Acc={intent_acc:.4f}")

    # Evaluate best checkpoint on test.
    best_model.to(device)
    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, best_model, lang
    )
    if "total" in results_test:
        writer.add_scalar("slot_f1/test", results_test["total"]["f"], cfg.n_epochs)
    writer.add_scalar("intent_acc/test", intent_test["accuracy"], cfg.n_epochs)
    writer.close()
    print(
        f"Best dev Slot F1: {best_f1:.4f} | Test Slot F1: {results_test['total']['f']:.4f} | "
        f"Test Intent Acc: {intent_test['accuracy']:.4f}"
    )
    return best_f1, results_test["total"]["f"], intent_test["accuracy"]
