from pathlib import Path

from functions import LoRAExperimentConfig, run_lora_experiment
from utils import PennTreeBank, build_loaders, build_tokenizer, read_file


def main():
    # Entry point for LoRA fine-tuning experiments.
    device = "cuda:0"
    # Resolve dataset paths relative to repo root.
    base_dir = Path(__file__).resolve().parents[2]

    train_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.test.txt"))

    # Wrap raw text into datasets for DataLoader.
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = build_tokenizer()
    # Build tokenized loaders with separate train/eval batch sizes.
    train_loader, dev_loader, test_loader = build_loaders(
        train_dataset,
        dev_dataset,
        test_dataset,
        tokenizer,
        device,
        train_batch_size=8,
        eval_batch_size=16,
    )

    # Define LoRA configurations to run.
    lora_experiments = [
        # LoRAExperimentConfig(name="LoRA r=8 a=16", rank=8, alpha=16, lr=5e-4),
        # LoRAExperimentConfig(name="LoRA r=4 a=16", rank=4, alpha=16, lr=5e-4),
        LoRAExperimentConfig(name="LoRA r=4 a=8", rank=4, alpha=8, lr=5e-4),
    ]

    lora_results = []
    # Run all configurations and collect metrics.
    for cfg in lora_experiments:
        lora_results.append((cfg.name, *run_lora_experiment(cfg, train_loader, dev_loader, test_loader, device, tokenizer)))

    print("\nSummary (name, best dev PPL, test PPL):")
    for r in lora_results:
        print(r)


if __name__ == "__main__":
    main()
