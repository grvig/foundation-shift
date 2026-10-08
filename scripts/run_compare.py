"""Paired comparison of every method in the main table against the reference method.

    python scripts/run_compare.py camelyon17

Reads results/<dataset>_main.csv and the predictions run_main saved under the data root,
and writes results/<dataset>_paired.csv. Nothing is refitted.
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
from src.evaluation.compare import compare_to_reference


class TestPredictions:
    """Loads a saved prediction file and keeps only the test rows."""

    def __init__(self, directory, test_mask):
        self.directory = directory
        self.test_mask = test_mask

    def __call__(self, key):
        path = self.directory / (key + ".npy")
        if not path.exists():
            raise FileNotFoundError("no saved predictions " + str(path)
                                    + "; run scripts/run_main.py first")
        return np.load(path)[self.test_mask]


def main():
    parser = argparse.ArgumentParser(description="Paired comparison against the reference.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.path("results_dir")
    table = pd.read_csv(results_dir / (args.dataset + "_main.csv"))
    metadata = load_metadata(config, args.dataset)
    test = (metadata["split"] == "test").to_numpy()
    bootstrap = config.section("bootstrap")
    settings = config.dataset(args.dataset)
    paired = compare_to_reference(
        table, TestPredictions(config.data_root() / "predictions" / args.dataset, test),
        metadata["label"].to_numpy()[test],
        metadata[settings["cluster_column"]].to_numpy()[test],
        settings["metric"], int(bootstrap["resamples"]),
        config.fresh_rng, config.section("comparison")["reference"])
    paired.to_csv(results_dir / (args.dataset + "_paired.csv"), index=False,
                  float_format="%.4f")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
