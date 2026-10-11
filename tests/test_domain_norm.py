import numpy as np
import pytest

from src.methods.domain_norm import PreparedScaler
from src.methods.domain_norm import standardise_by_domain
from src.methods.linear_probe import sweep


def test_every_domain_ends_up_centred_with_unit_spread():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 5)).astype(np.float16)
    domains = np.repeat([0, 1, 2], 100)
    features[domains == 2] = features[domains == 2] + 4.0
    out = standardise_by_domain(features, domains)
    assert out is features
    for domain in [0, 1, 2]:
        block = out[domains == domain].astype(np.float32)
        assert np.abs(block.mean(axis=0)).max() < 0.01
        assert block.std(axis=0) == pytest.approx(np.ones(5), abs=0.01)


def test_a_constant_feature_does_not_divide_by_zero():
    features = np.ones((10, 3), dtype=np.float32)
    features[:, 0] = np.arange(10)
    out = standardise_by_domain(features, np.zeros(10, dtype=int))
    assert np.isfinite(out).all()


def test_a_shifted_test_domain_is_recovered_by_its_own_statistics():
    """The test domain's features are offset; a train-fitted scaler would misplace
    them, per-domain statistics put them back where the classifier expects."""
    rng = np.random.default_rng(1)
    rows = []
    labels = []
    splits = []
    domains = []
    for split, domain, shift in [("train", 0, 0.0), ("id_val", 0, 0.0), ("ood_val", 1, 3.0),
                                 ("test", 2, 3.0)]:
        for _ in range(200):
            label = int(rng.integers(0, 2))
            vector = rng.normal(size=4)
            vector[0] = vector[0] + 2.0 * (2 * label - 1)
            vector[1] = vector[1] + 10.0
            vector[0] = vector[0] + shift
            rows.append(vector)
            labels.append(label)
            splits.append(split)
            domains.append(domain)
    features = standardise_by_domain(np.array(rows), np.array(domains))
    results, _ = sweep(features, np.array(labels), np.array(splits), [1.0], "accuracy",
                       500, 0, log=quiet, scaler=PreparedScaler())
    assert results[0]["test"] > 0.85


def quiet(message):
    pass
