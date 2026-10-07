import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_readme_table import replace_between
from make_readme_table import table_markdown

TABLE = pd.DataFrame([
    ["resnet50", "linear_probe", "ood_val", 0.9701, 0.9047, 0.8875, 0.8242, 0.9238],
    ["clip_b16", "zero_shot", "none", 0.6, 0.55, 0.512, 0.45, 0.6],
], columns=["backbone", "method", "selection", "id_val", "ood_val", "test", "test_low",
            "test_high"])


def test_rows_are_readable_percentages():
    text = table_markdown(TABLE)
    assert "| ResNet-50 (ImageNet) | linear probe (C on new-hospital val) | 97.0 | 90.5 | "\
           "**88.8** | 82.4 - 92.4 |" in text
    assert "| CLIP ViT-B/16 | zero-shot | 60.0 |" in text


def test_only_the_marked_block_changes():
    readme = "intro\n<!-- results:camelyon17 -->\nold\n<!-- /results:camelyon17 -->\noutro\n"
    updated = replace_between(readme, "camelyon17", "new")
    assert updated == ("intro\n<!-- results:camelyon17 -->\nnew\n"
                       "<!-- /results:camelyon17 -->\noutro\n")


def test_missing_markers_fail_loudly():
    with pytest.raises(ValueError):
        replace_between("no markers here", "camelyon17", "new")


def test_the_readme_has_markers_for_camelyon17():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "<!-- results:camelyon17 -->" in text
    assert "<!-- /results:camelyon17 -->" in text
