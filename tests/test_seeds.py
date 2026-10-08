import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("torch")

from run_finetune import replace_row
from summarise_seeds import summarise


def row(backbone, seed, test):
    return {"backbone": backbone, "seed": seed, "id_val": 0.9, "ood_val": 0.8, "test": test}


def test_a_rerun_replaces_only_its_own_backbone_and_seed(tmp_path):
    path = tmp_path / "seeds.csv"
    replace_row(path, row("clip_b16", 0, 0.90))
    replace_row(path, row("clip_b16", 1, 0.92))
    replace_row(path, row("resnet50", 0, 0.88))
    replace_row(path, row("clip_b16", 1, 0.93))
    table = pd.read_csv(path)
    assert len(table) == 3
    clip_seed_one = table[(table["backbone"] == "clip_b16") & (table["seed"] == 1)]
    assert list(clip_seed_one["test"]) == [0.93]


def test_files_from_before_seeds_existed_count_as_seed_zero(tmp_path):
    path = tmp_path / "old.csv"
    pd.DataFrame([{"backbone": "clip_b16", "test": 0.9}]).to_csv(path, index=False)
    replace_row(path, row("clip_b16", 0, 0.95))
    assert list(pd.read_csv(path)["test"]) == [0.95]


def test_the_summary_reports_spread_per_backbone():
    seeds = pd.DataFrame([row("clip_b16", 0, 0.90), row("clip_b16", 1, 0.94),
                          row("resnet50", 0, 0.88)])
    summary = summarise(seeds)
    clip = summary[summary["backbone"] == "clip_b16"].iloc[0]
    assert clip["seeds"] == 2
    assert clip["test_mean"] == pytest.approx(0.92)
    assert clip["test_min"] == pytest.approx(0.90)
    assert clip["test_sd"] == pytest.approx(0.0283, abs=1e-4)
    resnet = summary[summary["backbone"] == "resnet50"].iloc[0]
    assert pd.isna(resnet["test_sd"])
