import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.metrics import accuracy
from src.evaluation.metrics import macro_f1
from src.evaluation.metrics import metric_function
from src.evaluation.metrics import per_domain
from src.evaluation.metrics import worst_domain


def test_accuracy_by_hand():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_macro_f1_matches_the_wilds_definition():
    """WILDS passes labels=unique(y_true) to sklearn; ours must agree on random data."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = rng.integers(0, 12, size=300)
        predictions = rng.integers(0, 15, size=300)
        expected = f1_score(labels, predictions, average="macro", labels=np.unique(labels))
        assert macro_f1(labels, predictions) == pytest.approx(expected)


def test_a_predicted_class_absent_from_the_labels_is_not_averaged():
    # Class 9 is only ever predicted. Averaging over it would add a zero.
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 9]) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_perfect_predictions_score_one():
    assert macro_f1([3, 1, 2, 3], [3, 1, 2, 3]) == 1.0


def test_empty_inputs_fail_loudly():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        macro_f1([], [])


def test_per_domain_and_worst_domain():
    labels = [1, 1, 0, 0, 1, 1]
    predictions = [1, 1, 0, 1, 0, 0]
    domains = [0, 0, 3, 3, 4, 4]
    scores = per_domain(labels, predictions, domains, "accuracy")
    assert scores == {0: 1.0, 3: 0.5, 4: 0.0}
    assert worst_domain(labels, predictions, domains, "accuracy") == (4, 0.0)


def test_an_unknown_metric_fails_loudly():
    with pytest.raises(ValueError):
        metric_function("auc")
