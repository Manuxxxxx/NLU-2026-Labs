from pathlib import Path

from functions import ExperimentConfig, run_experiment
from utils import (
    IntentsAndSlots,
    Lang,
    build_loaders,
    create_dev_split,
    load_data,
)


def main():
    device = "cuda:0"
    pad_token = 0
    base_dir = Path(__file__).resolve().parents[2]

    tmp_train_raw = load_data(str(base_dir / "dataset/ATIS/train.json"))
    test_raw = load_data(str(base_dir / "dataset/ATIS/test.json"))

    train_raw, dev_raw, y_train, y_dev = create_dev_split(tmp_train_raw, portion=0.10, seed=42)

    y_test = [x["intent"] for x in test_raw]

    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0, pad_token=pad_token)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader, dev_loader, test_loader = build_loaders(
        train_dataset, dev_dataset, test_dataset, pad_token=pad_token, device=device
    )

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)

    experiments = [
        ExperimentConfig(name="Baseline", lr=1e-3),
        ExperimentConfig(name="Bigger d_model", d_model=32, ff_dim=64, lr=1e-3),
        ExperimentConfig(name="More heads+Bigger d", n_heads=2, d_model=32, ff_dim=64, lr=1e-3),
        ExperimentConfig(name="More layers+Bigger d", num_layers=2, d_model=32, ff_dim=64, lr=1e-3),
        ExperimentConfig(
            name="More layers+More heads+Bigger d", num_layers=2, n_heads=2, d_model=32, ff_dim=64, lr=5e-4
        ),
        ExperimentConfig(name="Out Dropout+Bigger d", d_model=32, ff_dim=64, out_dropout=0.1, lr=5e-4),
    ]

    results = []
    for cfg in experiments:
        results.append(
            (
                cfg.name,
                *run_experiment(
                    cfg,
                    train_loader,
                    dev_loader,
                    test_loader,
                    vocab_len,
                    slots_len,
                    n_intents,
                    device,
                    lang,
                ),
            )
        )

    print("\nSummary (name, best dev Slot F1, test Slot F1, test Intent Acc):")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
