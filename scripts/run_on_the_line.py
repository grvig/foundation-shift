"""Fit the ID-to-OOD line and score every method against it.

    python scripts/run_on_the_line.py camelyon17

Reads the probe sweep and the main table and writes two files to results/:
``<dataset>_on_the_line.csv`` has one row per point (backbone, method, C) with its
in-distribution accuracy, its test accuracy, the line's prediction and the effective
robustness; ``<dataset>_on_the_line_fit.csv`` holds the fitted slope, intercept and
correlation in probit space.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.config import load_config
from src.evaluation.on_the_line import effective_robustness
from src.evaluation.on_the_line import fit_line
from src.evaluation.on_the_line import predicted_ood


def collect_points(sweep, main):
    rows = []
    for _, row in sweep.iterrows():
        rows.append({"backbone": row["backbone"], "method": "linear_probe",
                     "c": row["c"], "id_val": row["id_val"], "test": row["test"]})
    for _, row in main[main["method"] == "zero_shot"].iterrows():
        rows.append({"backbone": row["backbone"], "method": "zero_shot",
                     "c": float("nan"), "id_val": row["id_val"], "test": row["test"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Accuracy on the line.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.path("results_dir")
    sweep = pd.read_csv(results_dir / (args.dataset + "_probe_sweep.csv"))
    main_table = pd.read_csv(results_dir / (args.dataset + "_main.csv"))
    points = collect_points(sweep, main_table)

    probes = points[points["method"] == "linear_probe"]
    slope, intercept, r = fit_line(probes["id_val"], probes["test"])
    points["predicted_test"] = predicted_ood(points["id_val"], slope, intercept)
    points["effective_robustness"] = effective_robustness(points["id_val"], points["test"],
                                                          slope, intercept)
    points.to_csv(results_dir / (args.dataset + "_on_the_line.csv"), index=False,
                  float_format="%.4f")
    fit = pd.DataFrame([{"points": len(probes), "slope": slope, "intercept": intercept,
                         "r": r}])
    fit.to_csv(results_dir / (args.dataset + "_on_the_line_fit.csv"), index=False,
               float_format="%.4f")
    print(points.to_string(index=False))
    print("line in probit space: slope " + format(slope, ".3f") + ", intercept "
          + format(intercept, ".3f") + ", r " + format(r, ".3f"))


if __name__ == "__main__":
    main()
