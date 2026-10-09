import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from per_slide import per_slide
from per_slide import worst_two_share


def test_each_held_out_slide_gets_a_row():
    labels = np.array([1, 0, 1, 1, 0, 0])
    predictions = np.array([1, 1, 1, 0, 0, 0])
    splits = np.array(["test", "test", "test", "test", "ood_val", "train"])
    clusters = np.array([3, 3, 4, 4, 7, 9])
    rows = per_slide(labels, predictions, splits, clusters)
    assert [(row["split"], row["cluster"]) for row in rows] == [("ood_val", 7), ("test", 3),
                                                               ("test", 4)]
    assert rows[1]["accuracy"] == 0.5
    assert rows[1]["positive_share"] == 0.5


def test_errors_concentrated_on_two_slides_are_measured():
    rows = [{"split": "test", "errors": 50}, {"split": "test", "errors": 30},
            {"split": "test", "errors": 10}, {"split": "test", "errors": 10},
            {"split": "ood_val", "errors": 999}]
    assert worst_two_share(rows) == pytest.approx(0.8)
    assert worst_two_share([{"split": "test", "errors": 0}]) == 0.0
