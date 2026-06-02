from pathlib import Path

from functions import ExperimentConfig, run_experiment
from utils import PennTreeBank, build_loaders, build_tokenizer, read_file


def main():
    device = "cuda:0"
    base_dir = Path(__file__).resolve().parents[2]

    train_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(str(base_dir / "dataset/PennTreeBank/ptb.test.txt"))

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = build_tokenizer()
    train_loader, dev_loader, test_loader = build_loaders(
        train_dataset,
        dev_dataset,
        test_dataset,
        tokenizer,
        device,
        train_batch_size=8,
        eval_batch_size=16,
    )

    vocab_len = len(tokenizer)

    experiments = [
        ExperimentConfig(name="wt + ff dropout 0.1 + more heads + moreke layers + bigger d_model (decreased lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, dropout=0.1, lr=0.005, save_best=True, weight_tying=True, n_epochs=20),
        ExperimentConfig(name="wt + ff dropout 0.1 + more heads + moreke layers + bigger d_model_2 (decreased lr)", d_model=64, n_heads=2, num_layers=2, ff_dim=128, dropout=0.1, lr=0.005, save_best=True, weight_tying=True, n_epochs=20),
        
        # ExperimentConfig(name="weight tying + more heads + more layers + bigger d_model (decreased lr, higher lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, lr=0.007, weight_tying=True, save_best=True),
        # ExperimentConfig(name="emb dropout 0.05 + more heads + more layers + bigger d_model (decreased lr, slightly higher lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.05, lr=0.005, save_best=False, n_epochs=20),
        # ExperimentConfig(name="attn dropout 0.05 + more heads + more layers + bigger d_model (decreased lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.05, lr=0.005, save_best=True),
        # ExperimentConfig(name="ff dropout 0.1 + more heads + more layers + bigger d_model (decreased lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="proj dropout 0.05 + more heads + more layers + bigger d_model (decreased lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, proj_dropout=0.05, lr=0.005, save_best=True),
        # Baseline (no dropout, no weight tying)
        # ExperimentConfig(name="Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, lr=0.01, save_best=True),

        # Hyperparameter-only (no dropout, no weight tying)
        # ExperimentConfig(name="bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, lr=0.01, save_best=True),
        # ExperimentConfig(name="more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, lr=0.01, save_best=True),
        # ExperimentConfig(name="more layers + bigger d_model (decreased lr)", d_model=32, n_heads=1, num_layers=2, ff_dim=64, lr=0.005, save_best=True),
        # ExperimentConfig(name="more heads + more layers + bigger d_model (decreased lr)", d_model=32, n_heads=2, num_layers=2, ff_dim=64, lr=0.005, save_best=True),

        # # Dropout combinations (no weight tying) - Baseline
        # ExperimentConfig(name="emb dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),

        # # Dropout combinations (no weight tying) - bigger d_model
        # ExperimentConfig(name="emb dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),

        # # Dropout combinations (no weight tying) - more heads + bigger d_model
        # ExperimentConfig(name="emb dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),

        # # Dropout combinations (no weight tying) - more layers + bigger d_model
        # ExperimentConfig(name="emb dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="attn dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, proj_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),

        # Dropout combinations (no weight tying) - more heads + more layers + bigger d_model
        # ExperimentConfig(name="emb dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="attn dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, proj_dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, dropout=0.1, lr=0.005, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="attn dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),
        # ExperimentConfig(name="emb dropout + attn dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, save_best=True),

        # # Weight tying only
        # ExperimentConfig(name="weight tying + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, lr=0.01, weight_tying=True, save_best=True),

        # # Weight tying + dropout combinations - Baseline
        # ExperimentConfig(name="weight tying + emb dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + ff dropout + Baseline", d_model=20, n_heads=1, num_layers=1, ff_dim=20, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),

        # # Weight tying + dropout combinations - bigger d_model
        # ExperimentConfig(name="weight tying + emb dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + ff dropout + bigger d_model", d_model=32, n_heads=1, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),

        # # Weight tying + dropout combinations - more heads + bigger d_model
        # ExperimentConfig(name="weight tying + emb dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + ff dropout + more heads + bigger d_model", d_model=32, n_heads=2, num_layers=1, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),

        # # Weight tying + dropout combinations - more layers + bigger d_model
        # ExperimentConfig(name="weight tying + emb dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + ff dropout + more layers + bigger d_model", d_model=32, n_heads=1, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),

        # # Weight tying + dropout combinations - more heads + more layers + bigger d_model
        # ExperimentConfig(name="weight tying + emb dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + attn dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
        # ExperimentConfig(name="weight tying + emb dropout + attn dropout + proj dropout + ff dropout + more heads + more layers + bigger d_model", d_model=32, n_heads=2, num_layers=2, ff_dim=64, emb_dropout=0.1, attn_dropout=0.1, proj_dropout=0.1, dropout=0.1, lr=0.01, weight_tying=True, save_best=True),
    ]

    results = []
    for cfg in experiments:
        results.append((cfg.name, *run_experiment(cfg, train_loader, dev_loader, test_loader, vocab_len, device, tokenizer)))

    print("\nSummary (name, best dev PPL, test PPL):")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
