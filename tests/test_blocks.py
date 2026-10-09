import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarise_blocks import combine
from src.plots.figures import plot_blocks


def run(backbone, method, test, seed=0):
    return {"backbone": backbone, "method": method, "seed": seed, "id_val": 0.97,
            "ood_val": 0.9, "test": test, "test_low": test - 0.03, "test_high": test + 0.02}


def test_configured_and_ablation_runs_are_joined_by_block_count(tmp_path):
    main = pd.DataFrame([run("clip_b16", "finetune_last2", 0.95),
                         run("resnet50", "finetune_last2", 0.91)])
    ablation = pd.DataFrame([run("clip_b16", "finetune_last1", 0.93),
                             run("clip_b16", "finetune_last4", 0.96),
                             run("clip_b16", "finetune_last4", 0.80, seed=1)])
    summary = combine(main, ablation)
    clip = summary[summary["backbone"] == "clip_b16"]
    assert list(clip["blocks"]) == [1, 2, 4]
    assert list(clip["test"]) == [0.93, 0.95, 0.96]
    plot_blocks(summary, tmp_path / "blocks")
    assert (tmp_path / "blocks.png").stat().st_size > 1000
