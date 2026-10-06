"""Confidence intervals that respect how the data was collected.

Camelyon17 patches are cut from whole tissue slides, thousands per slide, and patches from
one slide share its staining, its scanner settings and its patient. They are nowhere near
independent. Resampling individual patches would treat 85,000 test patches as 85,000
pieces of evidence when the test hospital really contributes ten slides, and the
intervals would come out far too narrow.

So the bootstrap resamples whole clusters (slides) with replacement and recomputes the
metric on every patch of the chosen slides. The interval then reflects how much the score
would move with a different draw of slides from the same hospital, which is the
uncertainty that matters for a claim about deployment.

The paired version resamples the same slides for both methods, so slide-to-slide
difficulty cancels and the interval is for the difference itself.
"""

import numpy as np

from src.evaluation.metrics import metric_function


def cluster_indices(clusters):
    """Row positions of every cluster, as a list of arrays."""
    clusters = np.asarray(clusters)
    groups = []
    for value in np.unique(clusters):
        groups.append(np.flatnonzero(clusters == value))
    return groups


def resample(groups, rng):
    chosen = rng.integers(0, len(groups), size=len(groups))
    parts = []
    for index in chosen:
        parts.append(groups[index])
    return np.concatenate(parts)


def cluster_bootstrap(labels, predictions, clusters, metric_name, resamples, rng):
    """Point estimate and 95% percentile interval for one method."""
    metric = metric_function(metric_name)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    groups = cluster_indices(clusters)
    values = np.empty(resamples)
    for step in range(resamples):
        rows = resample(groups, rng)
        values[step] = metric(labels[rows], predictions[rows])
    low, high = np.percentile(values, [2.5, 97.5])
    return metric(labels, predictions), float(low), float(high)


def paired_cluster_bootstrap(labels, first, second, clusters, metric_name, resamples, rng):
    """Difference first minus second, with both methods scored on the same resamples."""
    metric = metric_function(metric_name)
    labels = np.asarray(labels)
    first = np.asarray(first)
    second = np.asarray(second)
    groups = cluster_indices(clusters)
    values = np.empty(resamples)
    for step in range(resamples):
        rows = resample(groups, rng)
        values[step] = metric(labels[rows], first[rows]) - metric(labels[rows], second[rows])
    low, high = np.percentile(values, [2.5, 97.5])
    difference = metric(labels, first) - metric(labels, second)
    return difference, float(low), float(high)
