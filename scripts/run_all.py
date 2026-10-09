"""Regenerate every result, figure and README table for a dataset from its embeddings.

    python scripts/run_all.py camelyon17

Runs, in order: run_main (probes, zero-shot, WiSE-FT), run_compare (paired intervals
against the references), run_on_the_line, summarise_seeds, per_slide, data_card,
make_figures and make_readme_table.
Each step is its own script and can be rerun alone; this only saves typing them out.
Fine-tuning is not included because it needs the GPU for about twenty minutes per
backbone; run scripts/run_finetune.py for that.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STEPS = ["run_main.py", "run_compare.py", "run_on_the_line.py", "summarise_seeds.py",
         "per_slide.py",
         "data_card.py", "make_figures.py", "make_readme_table.py"]


def main():
    parser = argparse.ArgumentParser(description="Regenerate every result for a dataset.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    for step in STEPS:
        command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / step), args.dataset]
        if args.config is not None:
            command = command + ["--config", args.config]
        print("== " + step)
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT)
        if completed.returncode != 0:
            raise SystemExit(step + " failed with exit code " + str(completed.returncode))
    print("all steps finished")


if __name__ == "__main__":
    main()
