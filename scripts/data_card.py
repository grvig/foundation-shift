"""Write results/<dataset>_data_card.csv and print it.

    python scripts/data_card.py camelyon17

Needs only the metadata, not the images.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.data.card import summarise
from src.data.datasets import load_metadata


def main():
    parser = argparse.ArgumentParser(description="Summarise a dataset by split and domain.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    card = summarise(load_metadata(config, args.dataset))
    card.to_csv(config.path("results_dir") / (args.dataset + "_data_card.csv"),
                index=False, float_format="%.4f")
    print(card.to_string(index=False))
    print("total patches: " + str(int(card["patches"].sum())))


if __name__ == "__main__":
    main()
