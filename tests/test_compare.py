import numpy as np
import pandas as pd
import pytest

from src.evaluation.compare import compare_to_reference

REFERENCE = {"backbone": "resnet50", "method": "linear_probe", "selection": "ood_val"}
CLUSTERS = np.repeat(np.arange(10), 50)
LABELS = np.ones(500, dtype=int)


def predictions_with_errors(every):
    predictions = np.ones(500, dtype=int)
    predictions[::every] = 0
    return predictions


SAVED = {
    "resnet50_linear_probe_ood_val": predictions_with_errors(5),
    "clip_b16_linear_probe_ood_val": predictions_with_errors(50),
    "clip_b16_zero_shot_none": predictions_with_errors(5),
}


def load(key):
    return SAVED[key]


def table(rows):
    return pd.DataFrame(rows, columns=["backbone", "method", "selection", "test"])


def make_rng():
    return np.random.default_rng(0)


def test_a_clearly_better_method_is_significant_and_an_equal_one_is_not():
    result = compare_to_reference(table([
        ["resnet50", "linear_probe", "ood_val", 0.8],
        ["clip_b16", "linear_probe", "ood_val", 0.98],
        ["clip_b16", "zero_shot", "none", 0.8],
    ]), load, LABELS, CLUSTERS, "accuracy", 300, make_rng, REFERENCE)
    assert list(result["backbone"]) == ["clip_b16", "clip_b16"]
    better = result.iloc[0]
    assert better["difference"] == pytest.approx(0.18)
    assert bool(better["significant"])
    same = result.iloc[1]
    assert same["difference"] == 0.0
    assert not bool(same["significant"])


def test_the_reference_must_appear_exactly_once():
    with pytest.raises(ValueError):
        compare_to_reference(table([["clip_b16", "zero_shot", "none", 0.8]]), load, LABELS,
                             CLUSTERS, "accuracy", 10, make_rng, REFERENCE)
