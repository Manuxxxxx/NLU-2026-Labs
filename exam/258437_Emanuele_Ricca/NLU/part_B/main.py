from pathlib import Path

from functions import ExperimentConfigBert, ExperimentConfigGPT, run_experiment
from utils import Lang
from utils import load_data, create_dev_split


def main():
    device = "cuda:0"
    base_dir = Path(__file__).resolve().parents[2]

    tmp_train_raw = load_data(str(base_dir / "dataset/ATIS/train.json"))
    test_raw = load_data(str(base_dir / "dataset/ATIS/test.json"))
    train_raw, dev_raw, _, _ = create_dev_split(tmp_train_raw, portion=0.10, seed=42)

    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0, pad_token=0)

    experiments = [
        # ExperimentConfigBert(
        #     name="BERT-base",
        #     model_name="bert-base-uncased",
        #     lr=2e-5,
        #     batch_size=16,
        #     max_length=128,
        #     n_epochs=5,
        #     patience=5,
        #     warmup_epochs=10,
        #     dropout=0.1,
        # ),
        ExperimentConfigGPT(
            name="GPT2",
            model_name="openai-community/gpt2",
            lr=5e-5,
            batch_size=16,
            max_length=128,
            n_epochs=5,
            patience=5,
            warmup_epochs=10,
            dropout=0.1,
        ),
    ]

    results = []
    for cfg in experiments:
        results.append(run_experiment(cfg, train_raw, dev_raw, test_raw, lang, device))

    print("\nSummary (name, best dev Slot F1, test Slot F1, test Intent Acc):")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
