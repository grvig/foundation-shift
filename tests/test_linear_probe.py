import numpy as np
import pytest

from src.methods.linear_probe import LinearProbe
from src.methods.linear_probe import l2_normalise
from src.methods.linear_probe import select
from src.methods.linear_probe import sweep


def toy_data(seed=0):
    """Two classes separated along one axis, with a shifted domain for ood_val and test."""
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    splits = []
    for split, shift, count in [("train", 0.0, 400), ("id_val", 0.0, 100),
                                ("ood_val", 1.5, 100), ("test", 1.5, 100)]:
        for _ in range(count):
            label = int(rng.integers(0, 2))
            vector = rng.normal(0.0, 1.0, size=8)
            vector[0] = vector[0] + 3.0 * (2 * label - 1)
            vector[1] = vector[1] + 5.0 + shift
            features.append(vector)
            labels.append(label)
            splits.append(split)
    return np.array(features), np.array(labels), np.array(splits)


def test_l2_normalise_gives_unit_rows_and_survives_zero_rows():
    result = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert result[0] == pytest.approx([0.6, 0.8])
    assert list(result[1]) == [0.0, 0.0]


def test_a_separable_problem_is_learned():
    features, labels, splits = toy_data()
    train = splits == "train"
    probe = LinearProbe(1.0, 1000, 0).fit(features[train], labels[train])
    accuracy = np.mean(probe.predict(features[splits == "id_val"])
                       == labels[splits == "id_val"])
    assert accuracy > 0.9


def test_predicting_before_fitting_fails_loudly():
    with pytest.raises(RuntimeError):
        LinearProbe(1.0, 100, 0).predict(np.zeros((2, 3)))


def test_the_sweep_scores_every_c_on_every_split():
    features, labels, splits = toy_data()
    rows, probes = sweep(features, labels, splits, [0.01, 1.0], "accuracy", 1000, 0,
                         log=quiet)
    assert [row["c"] for row in rows] == [0.01, 1.0]
    assert set(probes.keys()) == {0.01, 1.0}
    for row in rows:
        for split in ["id_val", "ood_val", "test"]:
            assert 0.0 <= row[split] <= 1.0


def test_the_sweep_is_deterministic():
    features, labels, splits = toy_data()
    first, _ = sweep(features, labels, splits, [0.1], "accuracy", 1000, 0, log=quiet)
    second, _ = sweep(features, labels, splits, [0.1], "accuracy", 1000, 0, log=quiet)
    assert first == second


def test_selection_uses_the_named_split_and_breaks_ties_towards_small_c():
    rows = [{"c": 1.0, "id_val": 0.9, "ood_val": 0.7},
            {"c": 0.1, "id_val": 0.8, "ood_val": 0.7},
            {"c": 10.0, "id_val": 0.95, "ood_val": 0.6}]
    assert select(rows, "id_val")["c"] == 10.0
    assert select(rows, "ood_val")["c"] == 0.1


def quiet(message):
    pass
