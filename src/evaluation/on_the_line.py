"""Accuracy on the line, and effective robustness.

Across many models trained on the same data, accuracy on a shifted test set tends to be a
near-linear function of in-distribution accuracy once both are probit-transformed (the
inverse of the standard normal CDF). If that holds, most of what looks like robustness is
just being more accurate, and the interesting quantity is the vertical distance from the
line: **effective robustness**, the out-of-distribution accuracy a model has beyond what
its in-distribution accuracy predicts.

Zero-shot CLIP was reported to sit well above the line on natural-image ImageNet shifts.
Whether anything similar happens on a medical deployment shift is one of this project's
questions.

The line is fitted on the linear-probe points (every backbone at every regularisation
strength), then every point, including zero-shot ones, is scored against it. Probit space
is used because accuracies near 1 are compressed in linear space: the step from 0.98 to
0.99 is as hard as a much larger step in the middle of the range.
"""

import numpy as np
from scipy.stats import norm

EDGE = 1e-4


def probit(accuracy):
    return norm.ppf(np.clip(np.asarray(accuracy, dtype=float), EDGE, 1.0 - EDGE))


def fit_line(id_accuracy, ood_accuracy):
    """Least-squares line in probit space; returns (slope, intercept, pearson r)."""
    x = probit(id_accuracy)
    y = probit(ood_accuracy)
    if len(x) < 3:
        raise ValueError("need at least three points to fit and judge a line")
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r


def predicted_ood(id_accuracy, slope, intercept):
    return norm.cdf(intercept + slope * probit(id_accuracy))


def effective_robustness(id_accuracy, ood_accuracy, slope, intercept):
    """Measured OOD accuracy minus the line's prediction, in accuracy units."""
    return np.asarray(ood_accuracy, dtype=float) - predicted_ood(id_accuracy, slope, intercept)
