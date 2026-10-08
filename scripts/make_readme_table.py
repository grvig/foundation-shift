"""Write the results table into README.md from the CSVs, between marker comments.

    python scripts/make_readme_table.py camelyon17

The README never holds a hand-typed number: everything between
``<!-- results:<dataset> -->`` and ``<!-- /results:<dataset> -->`` is regenerated from
results/<dataset>_main.csv (and _finetune.csv when it exists).
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.config import load_config
from src.plots.figures import BACKBONE_LABELS

METHOD_LABELS = {"linear_probe": "linear probe", "zero_shot": "zero-shot",
                 "wise_ft": "WiSE-FT", "finetune_last2": "fine-tune last 2 blocks"}
SELECTION_LABELS = {"ood_val": "C on new-hospital val", "id_val": "C on familiar val",
                    "none": ""}


def method_label(row):
    method = METHOD_LABELS.get(row["method"], row["method"].replace("_", " "))
    selection = SELECTION_LABELS.get(row["selection"], row["selection"])
    if selection == "":
        return method
    return method + " (" + selection + ")"


def percent(value):
    return format(100.0 * value, ".1f")


def table_markdown(table):
    lines = ["| Backbone | Method | Familiar hospitals | New hospital (val) | "
             "New hospital (test) | Test 95% interval |",
             "|---|---|---|---|---|---|"]
    for _, row in table.iterrows():
        lines.append("| " + BACKBONE_LABELS.get(row["backbone"], row["backbone"]) + " | "
                     + method_label(row) + " | " + percent(row["id_val"]) + " | "
                     + percent(row["ood_val"]) + " | **" + percent(row["test"]) + "** | "
                     + percent(row["test_low"]) + " - " + percent(row["test_high"]) + " |")
    return "\n".join(lines)


def replace_between(text, dataset, block):
    start = "<!-- results:" + dataset + " -->"
    end = "<!-- /results:" + dataset + " -->"
    if start not in text or end not in text:
        raise ValueError("README has no " + start + " ... " + end + " markers")
    before = text.split(start)[0]
    after = text.split(end)[1]
    return before + start + "\n" + block + "\n" + end + after


def main():
    parser = argparse.ArgumentParser(description="Regenerate the README results table.")
    parser.add_argument("dataset")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results = config.path("results_dir")
    table = pd.read_csv(results / (args.dataset + "_main.csv"))
    finetune_path = results / (args.dataset + "_finetune.csv")
    if finetune_path.exists():
        table = pd.concat([table, pd.read_csv(finetune_path)], ignore_index=True)
    readme = config.project_root / "README.md"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(replace_between(text, args.dataset, table_markdown(table)),
                      encoding="utf-8")
    print("README table for " + args.dataset + " regenerated: " + str(len(table)) + " rows")


if __name__ == "__main__":
    main()
