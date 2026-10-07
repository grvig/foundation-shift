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


def main():
    parser = argparse.ArgumentParser(description="Draw the report figures.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    results = config.path("results_dir")
    figures = config.path("figures_dir")
    plot_main(pd.read_csv(results / (args.dataset + "_main.csv")),
              figures / (args.dataset + "_main"))
    line_path = results / (args.dataset + "_on_the_line.csv")
    if line_path.exists():
        fit = pd.read_csv(results / (args.dataset + "_on_the_line_fit.csv")).iloc[0]
        plot_on_the_line(pd.read_csv(line_path), fit,
                         figures / (args.dataset + "_on_the_line"))
    print("figures written to " + str(figures))


if __name__ == "__main__":
    main()
