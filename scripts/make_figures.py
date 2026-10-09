"""Draw every report figure for a dataset from its CSVs in results/.

    python scripts/make_figures.py camelyon17

Writes PDF (for the report) and PNG (for the README) versions to figures/.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.config import load_config
from src.plots.figures import plot_main
from src.plots.figures import plot_on_the_line
from src.plots.figures import plot_paired


def main():
    parser = argparse.ArgumentParser(description="Draw the report figures.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    results = config.path("results_dir")
    figures = config.path("figures_dir")
    main_table = pd.read_csv(results / (args.dataset + "_main.csv"))
    finetune_path = results / (args.dataset + "_finetune.csv")
    if finetune_path.exists():
        main_table = pd.concat([main_table, pd.read_csv(finetune_path)], ignore_index=True)
    plot_main(main_table, figures / (args.dataset + "_main"))
    paired_path = results / (args.dataset + "_paired.csv")
    if paired_path.exists():
        plot_paired(pd.read_csv(paired_path), figures / (args.dataset + "_paired"))
    line_path = results / (args.dataset + "_on_the_line.csv")
    if line_path.exists():
        fit = pd.read_csv(results / (args.dataset + "_on_the_line_fit.csv")).iloc[0]
        plot_on_the_line(pd.read_csv(line_path), fit,
                         figures / (args.dataset + "_on_the_line"))
    print("figures written to " + str(figures))


if __name__ == "__main__":
    main()
