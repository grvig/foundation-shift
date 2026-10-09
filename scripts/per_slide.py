"""Accuracy of every saved method on every slide of the held-out hospitals.

    python scripts/per_slide.py camelyon17

Writes results/<dataset>_per_slide.csv: one row per (method, split, slide) with the
number of patches, the tumour share and the accuracy. If errors pile up on a few slides,
the test set holds far fewer independent pieces of evidence than its patch count
suggests, which is the reason every interval in this project resamples slides.

Also prints, per method, the share of all its test errors that fall on its two worst
slides.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.datasets import load_metadata

SPLITS = ["ood_val", "test"]


def per_slide(labels, predictions, splits, clusters):
    rows = []
    for split in SPLITS:
        in_split = splits == split
        for cluster in np.unique(clusters[in_split]):
            mask = in_split & (clusters == cluster)
            rows.append({"split": split, "cluster": cluster.item(),
                         "images": int(mask.sum()),
                         "positive_share": float(labels[mask].mean()),
                         "accuracy": float(np.mean(predictions[mask] == labels[mask])),
                         "errors": int(np.sum(predictions[mask] != labels[mask]))})
    return rows


def worst_two_share(rows):
    """Share of test errors on the two clusters with the most errors."""
    test = []
    for row in rows:
        if row["split"] == "test":
            test.append(row["errors"])
    test = sorted(test, reverse=True)
    total = sum(test)
    if total == 0:
        return 0.0
    return sum(test[:2]) / total


def main():
    parser = argparse.ArgumentParser(description="Accuracy slide by slide.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.dataset(args.dataset)
    metadata = load_metadata(config, args.dataset)
    labels = metadata["label"].to_numpy()
    splits = metadata["split"].to_numpy()
    clusters = metadata[settings["cluster_column"]].to_numpy()
    directory = config.data_root() / "predictions" / args.dataset
    output = []
    for path in sorted(directory.glob("*.npy")):
        if "_seed" in path.stem:
            continue
        rows = per_slide(labels, np.load(path), splits, clusters)
        for row in rows:
            output.append(dict({"method": path.stem}, **row))
        print(path.stem + ": " + format(100 * worst_two_share(rows), ".0f")
              + "% of test errors on the two worst slides")
    pd.DataFrame(output).to_csv(config.path("results_dir") / (args.dataset + "_per_slide.csv"),
                                index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
