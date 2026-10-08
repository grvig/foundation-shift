"""The main comparison: every backbone, linear probe, zero-shot and WiSE-FT, ID versus OOD.

    python scripts/run_main.py camelyon17

Writes two CSVs to results/:

``<dataset>_main.csv``
    One row per backbone and method: the in-distribution score (id_val), the score on
    the validation hospital (ood_val) and the score on the test hospital (test). Linear
    probes appear twice, once with C chosen on ood_val (the WILDS rule) and once with C
    chosen on id_val (what a deployment without data from the new site could do).
    test_low and test_high are a 95% interval from resampling whole slides.
``<dataset>_probe_sweep.csv``
    Every C tried for every backbone, so the selection can be audited.
``<dataset>_wise_ft_sweep.csv``
    Every alpha tried for each image-text backbone and selection rule.
``<dataset>_per_domain.csv``
    Every method's score inside each hospital on each split. The pooled in-distribution
    score is dominated by the largest training hospital; this shows the smallest too.

Backbones without finished embeddings are skipped with a message rather than failing,
so the table can be filled in one backbone at a time.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.datasets import load_metadata
from src.embeddings.store import load_embeddings
from src.evaluation.bootstrap import cluster_bootstrap
from src.evaluation.metrics import metric_function
from src.evaluation.metrics import per_domain
from src.methods.linear_probe import EVALUATION_SPLITS
from src.methods.linear_probe import select
from src.methods.linear_probe import sweep
from src.methods.wise_ft import alpha_sweep
from src.methods.wise_ft import blend
from src.methods.wise_ft import select_alpha
from src.methods.zero_shot import load_text_head
from src.methods.zero_shot import zero_shot_predict
from src.models.backbones import backbone_names
from src.models.backbones import backbone_settings
from src.models.backbones import supports_text


def score_row(context, backbone, method, selection, c, predictions):
    """One results row: every split's score plus a slide-clustered interval on test.

    The predictions are also saved under the data root so later scripts can compare
    methods on identical resamples without refitting anything.
    """
    metric = metric_function(context["metric"])
    labels = context["labels"]
    splits = context["splits"]
    row = {"dataset": context["dataset"], "backbone": backbone, "method": method,
           "selection": selection, "c": c}
    for split in EVALUATION_SPLITS:
        mask = splits == split
        row[split] = metric(labels[mask], predictions[mask])
    test = splits == "test"
    _, low, high = cluster_bootstrap(labels[test], predictions[test],
                                     context["clusters"][test], context["metric"],
                                     context["resamples"], context["config"].fresh_rng())
    row["test_low"] = low
    row["test_high"] = high
    for split in EVALUATION_SPLITS:
        mask = splits == split
        scores = per_domain(labels[mask], predictions[mask], context["domains"][mask],
                            context["metric"])
        for domain in sorted(scores.keys()):
            context["domain_rows"].append({"backbone": backbone, "method": method,
                                           "selection": selection, "split": split,
                                           "domain": domain, "score": scores[domain]})
    directory = context["predictions_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / (backbone + "_" + method + "_" + selection + ".npy"),
            predictions.astype(np.int16))
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
    bootstrap = config.section("bootstrap")
    context = {"config": config, "dataset": args.dataset, "metric": settings["metric"],
               "labels": labels, "splits": splits,
               "clusters": metadata[bootstrap["cluster_column"]].to_numpy(),
               "resamples": int(bootstrap["resamples"]),
               "domains": metadata[settings["domain_column"]].to_numpy(),
               "domain_rows": [],
               "predictions_dir": config.data_root() / "predictions" / args.dataset}

    main_rows = []
    sweep_rows = []
    wise_rows = []
    for backbone in backbone_names(config):
        try:
            features = load_embeddings(config.data_root(), args.dataset, backbone, paths,
                                       backbone_settings(config, backbone))
        except FileNotFoundError as error:
            print("skipping " + backbone + ": " + str(error))
            continue
        print(backbone + ": linear probe")
        rows, probes = sweep(features, labels, splits, probe["c_grid"], settings["metric"],
                             probe["max_iter"], config.seed)
        for row in rows:
            sweep_rows.append(dict({"dataset": args.dataset, "backbone": backbone}, **row))
        for selection in probe["selection_splits"]:
            c = select(rows, selection)["c"]
            predictions = probes[c].predict(features)
            main_rows.append(score_row(context, backbone, "linear_probe", selection, c,
                                       predictions))
        if supports_text(config, backbone):
            print(backbone + ": zero-shot and WiSE-FT")
            head = load_text_head(config, backbone, args.dataset, torch.device("cpu"))
            main_rows.append(score_row(context, backbone, "zero_shot", "none",
                                       float("nan"),
                                       zero_shot_predict(features, head.vectors)))
            zero_shot_logits = head.logits(features)
            for selection in probe["selection_splits"]:
                c = select(rows, selection)["c"]
                probe_logits = probes[c].logits(features)
                alpha_rows = alpha_sweep(zero_shot_logits, probe_logits, labels, splits,
                                         config.section("wise_ft")["alphas"],
                                         settings["metric"])
                for alpha_row in alpha_rows:
                    wise_rows.append(dict({"backbone": backbone, "selection": selection,
                                           "c": c}, **alpha_row))
                alpha = select_alpha(alpha_rows, selection)["alpha"]
                predictions = np.argmax(blend(zero_shot_logits, probe_logits, alpha), axis=1)
                row = score_row(context, backbone, "wise_ft", selection, c, predictions)
                row["alpha"] = alpha
                main_rows.append(row)

    if len(main_rows) == 0:
        raise SystemExit("no embeddings found; run scripts/embed.py first")
    columns = (["dataset", "backbone", "method", "selection", "c", "alpha"]
               + EVALUATION_SPLITS + ["test_low", "test_high"])
    table = pd.DataFrame(main_rows).reindex(columns=columns)
    results_dir = config.path("results_dir")
    table.to_csv(results_dir / (args.dataset + "_main.csv"), index=False,
                 float_format="%.4f")
    pd.DataFrame(sweep_rows).to_csv(results_dir / (args.dataset + "_probe_sweep.csv"),
                                    index=False, float_format="%.4f")
    if len(wise_rows) > 0:
        pd.DataFrame(wise_rows).to_csv(results_dir / (args.dataset + "_wise_ft_sweep.csv"),
                                       index=False, float_format="%.4f")
    pd.DataFrame(context["domain_rows"]).to_csv(
        results_dir / (args.dataset + "_per_domain.csv"), index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=format_score))


def format_score(value):
    return format(value, ".4f")


if __name__ == "__main__":
    main()
