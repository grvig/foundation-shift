"""Linear probes on per-hospital standardised embeddings (see src/methods/domain_norm.py).

    python scripts/run_domain_norm.py camelyon17
    python scripts/run_domain_norm.py camelyon17 --backbones clip_b16 siglip_b16

Writes results/<dataset>_domain_norm.csv in the main table's columns, with the method
``probe_domain_norm`` and both selection rules, and saves predictions next to the others
so run_compare.py can pair them with the ordinary probes. Rows for backbones not run
this time are kept, so the table can be filled one backbone at a time.
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for folder in [REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import pandas as pd

from run_main import score_row
from src.config import load_config
from src.data.datasets import load_metadata
from src.embeddings.store import load_embeddings
from src.methods.domain_norm import PreparedScaler
from src.methods.domain_norm import standardise_by_domain
from src.methods.linear_probe import EVALUATION_SPLITS
from src.methods.linear_probe import select
from src.methods.linear_probe import sweep
from src.models.backbones import backbone_names
from src.models.backbones import backbone_settings

METHOD = "probe_domain_norm"


def main():
    parser = argparse.ArgumentParser(description="Probes on per-domain standardised features.")
    parser.add_argument("dataset")
    parser.add_argument("--backbones", nargs="*", default=None)
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
    domains = metadata[settings["domain_column"]].to_numpy()
    context = {"config": config, "dataset": args.dataset, "metric": settings["metric"],
               "labels": labels, "splits": splits,
               "clusters": metadata[settings["cluster_column"]].to_numpy(),
               "resamples": int(config.section("bootstrap")["resamples"]),
               "domains": domains, "domain_rows": [],
               "predictions_dir": config.data_root() / "predictions" / args.dataset}

    names = args.backbones
    if names is None or len(names) == 0:
        names = backbone_names(config)
    rows = []
    for backbone in names:
        features = load_embeddings(config.data_root(), args.dataset, backbone, paths,
                                   backbone_settings(config, backbone), as_float32=False)
        standardise_by_domain(features, domains)
        print(backbone + ": probes on per-hospital standardised features")
        sweep_rows, probes = sweep(features, labels, splits, probe["c_grid"],
                                   settings["metric"], probe["max_iter"], config.seed,
                                   scaler=PreparedScaler())
        for selection in probe["selection_splits"]:
            c = select(sweep_rows, selection)["c"]
            rows.append(score_row(context, backbone, METHOD, selection, c,
                                  probes[c].predict(features)))
        del features

    path = config.path("results_dir") / (args.dataset + "_domain_norm.csv")
    table = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_csv(path)
        table = pd.concat([previous[~previous["backbone"].isin(names)], table],
                          ignore_index=True)
    columns = (["dataset", "backbone", "method", "selection", "c"] + EVALUATION_SPLITS
               + ["test_low", "test_high"])
    table = table.reindex(columns=columns)
    table.to_csv(path, index=False, float_format="%.4f")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
