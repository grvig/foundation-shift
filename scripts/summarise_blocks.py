"""How much of the model to fine-tune: 1, 2 or 4 final blocks, side by side.

    python scripts/summarise_blocks.py camelyon17

Joins the configured runs (results/<dataset>_finetune.csv, two blocks) with the ablation
runs (results/<dataset>_finetune_blocks.csv) into results/<dataset>_blocks_summary.csv,
one row per backbone and block count, and draws figures/<dataset>_blocks. All rows are
seed 0, so they differ only in how many blocks trained. Skips quietly when no ablation
has been run.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.config import load_config
from src.plots.figures import plot_blocks

COLUMNS = ["backbone", "blocks", "id_val", "ood_val", "test", "test_low", "test_high"]


def block_count(method):
    return int(method.replace("finetune_last", ""))


def combine(main, ablation):
    rows = []
    for table in [main, ablation]:
        for _, row in table.iterrows():
            if int(row.get("seed", 0)) != 0:
                continue
            entry = {"blocks": block_count(row["method"])}
            for column in COLUMNS:
                if column != "blocks":
                    entry[column] = row[column]
            rows.append(entry)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values(["backbone", "blocks"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Summarise the block-count ablation.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = config.path("results_dir")
    ablation_path = results / (args.dataset + "_finetune_blocks.csv")
    main_path = results / (args.dataset + "_finetune.csv")
    if not ablation_path.exists() or not main_path.exists():
        print("no block-count ablation recorded yet; nothing to summarise")
        return
    summary = combine(pd.read_csv(main_path), pd.read_csv(ablation_path))
    summary.to_csv(results / (args.dataset + "_blocks_summary.csv"), index=False,
                   float_format="%.4f")
    config.ensure_output_dirs()
    plot_blocks(summary, config.path("figures_dir") / (args.dataset + "_blocks"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
