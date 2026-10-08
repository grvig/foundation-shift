import numpy as np
import pytest

from src.methods.linear_probe import LinearProbe
from src.methods.wise_ft import alpha_sweep
from src.methods.wise_ft import blend
from src.methods.wise_ft import select_alpha
from src.methods.zero_shot import TextHead
from src.methods.zero_shot import zero_shot_predict


def test_the_ends_of_the_blend_are_the_two_classifiers():
    zero_shot = np.array([[1.0, 0.0], [0.0, 2.0]])
    probe = np.array([[0.0, 3.0], [5.0, 0.0]])
    assert np.array_equal(blend(zero_shot, probe, 0.0), zero_shot)
    assert np.array_equal(blend(zero_shot, probe, 1.0), probe)
    assert np.array_equal(blend(zero_shot, probe, 0.5), [[0.5, 1.5], [2.5, 1.0]])


def test_scale_and_bias_do_not_change_zero_shot_predictions():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3, 8))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    features = rng.normal(size=(50, 8))
    head = TextHead(vectors, scale=100.0, bias=-10.0)
    assert np.array_equal(np.argmax(head.logits(features), axis=1),
                          zero_shot_predict(features, vectors))


def test_binary_probe_logits_reproduce_its_predictions():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 4))
    labels = (features[:, 0] > 0).astype(int)
    probe = LinearProbe(1.0, 500, 0).fit(features, labels)
    logits = probe.logits(features)
    assert logits.shape == (200, 2)
    assert np.array_equal(np.argmax(logits, axis=1), probe.predict(features))


def test_each_split_selects_the_alpha_that_suits_it():
    """Zero-shot is right on id_val, the probe on ood_val. On the ood_val row zero-shot
    is confidently wrong, so only alpha = 1 gets it right."""
    labels = np.array([0, 1, 0, 1])
    splits = np.array(["id_val", "id_val", "ood_val", "test"])
    zero_shot = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 3.0], [1.0, 0.0]])
    probe = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    rows = alpha_sweep(zero_shot, probe, labels, splits, [0.0, 0.5, 1.0], "accuracy")
    assert [row["alpha"] for row in rows] == [0.0, 0.5, 1.0]
    assert select_alpha(rows, "id_val")["alpha"] == 0.0
    assert select_alpha(rows, "ood_val")["alpha"] == 1.0


def test_ties_keep_the_smaller_alpha():
    rows = [{"alpha": 0.2, "ood_val": 0.9}, {"alpha": 0.6, "ood_val": 0.9}]
    assert select_alpha(rows, "ood_val")["alpha"] == pytest.approx(0.2)
