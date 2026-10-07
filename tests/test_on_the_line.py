import numpy as np
import pytest
from scipy.stats import norm

from src.evaluation.on_the_line import effective_robustness
from src.evaluation.on_the_line import fit_line
from src.evaluation.on_the_line import predicted_ood
from src.evaluation.on_the_line import probit


def points_on_a_line(slope, intercept):
    id_accuracy = np.array([0.80, 0.85, 0.90, 0.95, 0.97])
    ood_accuracy = norm.cdf(intercept + slope * norm.ppf(id_accuracy))
    return id_accuracy, ood_accuracy


def test_a_probit_line_is_recovered_exactly():
    id_accuracy, ood_accuracy = points_on_a_line(0.7, -0.2)
    slope, intercept, r = fit_line(id_accuracy, ood_accuracy)
    assert slope == pytest.approx(0.7)
    assert intercept == pytest.approx(-0.2)
    assert r == pytest.approx(1.0)


def test_points_on_the_line_have_no_effective_robustness():
    id_accuracy, ood_accuracy = points_on_a_line(0.7, -0.2)
    robustness = effective_robustness(id_accuracy, ood_accuracy, 0.7, -0.2)
    assert np.allclose(robustness, 0.0)


def test_a_point_above_the_line_is_positive():
    expected = float(predicted_ood([0.9], 0.7, -0.2)[0])
    assert effective_robustness([0.9], [expected + 0.05], 0.7, -0.2)[0] == pytest.approx(0.05)


def test_perfect_accuracy_does_not_become_infinite():
    assert np.isfinite(probit([0.0, 1.0])).all()


def test_too_few_points_fail_loudly():
    with pytest.raises(ValueError):
        fit_line([0.9, 0.8], [0.7, 0.6])
