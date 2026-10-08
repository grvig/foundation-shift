"""Spread of fine-tuning results across seeds, per backbone.

    python scripts/summarise_seeds.py camelyon17

Reads results/<dataset>_finetune_seeds.csv and writes
results/<dataset>_finetune_seed_summary.csv with the number of seeds and the mean,
standard deviation, minimum and maximum of each split's score. The slide bootstrap says
how much a score would move with different test slides; this says how much it moves
with a different training run, which the bootstrap cannot see.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.config import load_config
from src.methods.linear_probe import EVALUATION_SPLITS


def summarise(seeds):
    rows = []
    for backbone in seeds["backbone"].unique():
        group = seeds[seeds["backbone"] == backbone]
        row = {"backbone": backbone, "seeds": len(group)}
        for split in EVALUATION_SPLITS:
            row[split + "_mean"] = float(group[split].mean())
            # Sample standard deviation; undefined (NaN) until there are two seeds.
            row[split + "_sd"] = float(group[split].std(ddof=1))
            row[split + "_min"] = float(group[split].min())
            row[split + "_max"] = float(group[split].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarise fine-tuning across seeds.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = config.path("results_dir")
    path = results / (args.dataset + "_finetune_seeds.csv")
    if not path.exists():
        print("no fine-tuning seeds recorded yet; nothing to summarise")
        return
    summary = summarise(pd.read_csv(path))
    summary.to_csv(results / (args.dataset + "_finetune_seed_summary.csv"), index=False,
                   float_format="%.4f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
