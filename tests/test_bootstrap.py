import numpy as np
import pytest

from src.evaluation.bootstrap import cluster_bootstrap
from src.evaluation.bootstrap import cluster_indices
from src.evaluation.bootstrap import paired_cluster_bootstrap


def test_cluster_indices_group_rows_by_cluster():
    groups = cluster_indices([5, 1, 5, 1, 9])
    assert [list(group) for group in groups] == [[1, 3], [0, 2], [4]]


def test_the_point_estimate_is_the_plain_metric():
    labels = [1, 0, 1, 1]
    predictions = [1, 0, 0, 1]
    estimate, low, high = cluster_bootstrap(labels, predictions, [0, 0, 1, 1], "accuracy",
                                            200, np.random.default_rng(0))
    assert estimate == 0.75
    assert low <= estimate <= high


def test_clustered_intervals_are_wider_than_row_intervals_when_slides_differ():
    """Ten slides, each either all right or all wrong: the evidence is ten slides, not
    ten thousand patches, and the interval must say so."""
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(10), 1000)
    labels = np.ones(len(clusters), dtype=int)
    predictions = np.where(clusters < 7, 1, 0)
    _, low, high = cluster_bootstrap(labels, predictions, clusters, "accuracy", 500, rng)
    _, row_low, row_high = cluster_bootstrap(labels, predictions, np.arange(len(clusters)),
                                             "accuracy", 100, np.random.default_rng(1))
    assert high - low > 0.3
    assert row_high - row_low < 0.05


def test_paired_difference_cancels_shared_difficulty():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(10), 100)
    labels = np.ones(1000, dtype=int)
    # Both methods fail on the same hard slides; the second also fails one row per slide.
    first = np.where(clusters < 5, 1, 0)
    second = first.copy()
    second[::100] = 0
    difference, low, high = paired_cluster_bootstrap(labels, first, second, clusters,
                                                     "accuracy", 500, rng)
    assert difference == pytest.approx(0.005)
    assert low >= 0.0
    assert high - low < 0.01


def test_the_bootstrap_is_reproducible_from_its_generator():
    args = ([1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1], [0, 0, 1, 1, 2, 2], "accuracy", 100)
    first = cluster_bootstrap(*args, np.random.default_rng(3))
    second = cluster_bootstrap(*args, np.random.default_rng(3))
    assert first == second
