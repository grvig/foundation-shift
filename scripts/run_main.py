"""The main comparison: every backbone, linear probe and zero-shot, ID versus OOD.

    python scripts/run_main.py camelyon17

Writes two CSVs to results/:

``<dataset>_main.csv``
    One row per backbone and method: the in-distribution score (id_val), the score on
    the validation hospital (ood_val) and the score on the test hospital (test). Linear
    probes appear twice, once with C chosen on ood_val (the WILDS rule) and once with C
    chosen on id_val (what a deployment without data from the new site could do).
``<dataset>_probe_sweep.csv``
    Every C tried for every backbone, so the selection can be audited.

Backbones without finished embeddings are skipped with a message rather than failing,
so the table can be filled in one backbone at a time.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
import torch

from src.config import load_config
from src.data.datasets import load_metadata
from src.embeddings.store import load_embeddings
from src.evaluation.metrics import metric_function
from src.methods.linear_probe import EVALUATION_SPLITS
from src.methods.linear_probe import select
from src.methods.linear_probe import sweep
from src.methods.zero_shot import load_text_side
from src.methods.zero_shot import zero_shot_predict
from src.models.backbones import backbone_names
from src.models.backbones import backbone_settings
from src.models.backbones import supports_text


def zero_shot_row(config, dataset, backbone, features, labels, splits, metric_name):
    metric = metric_function(metric_name)
    text = load_text_side(config, backbone, dataset, torch.device("cpu"))
    predictions = zero_shot_predict(features, text)
    row = {"dataset": dataset, "backbone": backbone, "method": "zero_shot",
           "selection": "none", "c": float("nan")}
    for split in EVALUATION_SPLITS:
        mask = splits == split
        row[split] = metric(labels[mask], predictions[mask])
    return row


def main():
    parser = argparse.ArgumentParser(description="Linear probe and zero-shot, every backbone.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    settings = config.dataset(args.dataset)
    probe = config.section("linear_probe")
    metadata = load_metadata(config, args.dataset)
    paths = list(metadata["path"])
    labels = metadata["label"].to_numpy()
    splits = metadata["split"].to_numpy()

    main_rows = []
    sweep_rows = []
    for backbone in backbone_names(config):
        try:
            features = load_embeddings(config.data_root(), args.dataset, backbone, paths,
                                       backbone_settings(config, backbone))
        except FileNotFoundError as error:
            print("skipping " + backbone + ": " + str(error))
            continue
        print(backbone + ": linear probe")
        rows, _ = sweep(features, labels, splits, probe["c_grid"], settings["metric"],
                        probe["max_iter"], config.seed)
        for row in rows:
            sweep_rows.append(dict({"dataset": args.dataset, "backbone": backbone}, **row))
        for selection in probe["selection_splits"]:
            chosen = select(rows, selection)
            main_rows.append(dict({"dataset": args.dataset, "backbone": backbone,
                                   "method": "linear_probe", "selection": selection},
                                  **chosen))
        if supports_text(config, backbone):
            print(backbone + ": zero-shot")
            main_rows.append(zero_shot_row(config, args.dataset, backbone, features, labels,
                                           splits, settings["metric"]))

    if len(main_rows) == 0:
        raise SystemExit("no embeddings found; run scripts/embed.py first")
    columns = ["dataset", "backbone", "method", "selection", "c"] + EVALUATION_SPLITS
    table = pd.DataFrame(main_rows)[columns]
    results_dir = config.path("results_dir")
    table.to_csv(results_dir / (args.dataset + "_main.csv"), index=False,
                 float_format="%.4f")
    pd.DataFrame(sweep_rows).to_csv(results_dir / (args.dataset + "_probe_sweep.csv"),
                                    index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=format_score))


def format_score(value):
    return format(value, ".4f")


if __name__ == "__main__":
    main()
