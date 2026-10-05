"""Metrics, defined the way WILDS defines them so results compare with its leaderboard.

Accuracy is the official Camelyon17 metric. Macro-F1 is the official iWildCam metric, and
WILDS averages it over the classes **present in the true labels** only. A species that
never appears in a test split adds no zero to the average; predicting it by mistake still
costs recall on the species that was really there. Averaging over every class instead
would make our numbers lower than the leaderboard's for a reason unrelated to the model.

Per-domain accuracy (one hospital, one camera) is reported alongside the pooled number,
because a model can look fine pooled while failing completely on one domain, and the worst
domain is what a deployment would actually experience.
"""

import numpy as np


def accuracy(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if len(labels) == 0:
        raise ValueError("accuracy of an empty set is undefined")
    return float(np.mean(labels == predictions))


def macro_f1(labels, predictions):
    """F1 averaged over the classes that occur in ``labels``."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if len(labels) == 0:
        raise ValueError("macro-F1 of an empty set is undefined")
    scores = []
    for value in np.unique(labels):
        true_positive = float(np.sum((labels == value) & (predictions == value)))
        predicted = float(np.sum(predictions == value))
        actual = float(np.sum(labels == value))
        if true_positive == 0.0:
            scores.append(0.0)
            continue
        precision = true_positive / predicted
        recall = true_positive / actual
        scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


METRICS = {"accuracy": accuracy, "macro_f1": macro_f1}


def metric_function(name):
    if name not in METRICS:
        raise ValueError("unknown metric: " + name)
    return METRICS[name]


def per_domain(labels, predictions, domains, metric_name):
    """The metric computed separately inside every domain, as {domain: value}."""
    function = metric_function(metric_name)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    domains = np.asarray(domains)
    results = {}
    for domain in np.unique(domains):
        mask = domains == domain
        results[domain.item()] = function(labels[mask], predictions[mask])
    return results


def worst_domain(labels, predictions, domains, metric_name):
    """The lowest per-domain score and which domain it came from."""
    scores = per_domain(labels, predictions, domains, metric_name)
    worst = None
    for domain in sorted(scores.keys()):
        if worst is None or scores[domain] < scores[worst]:
            worst = domain
    return worst, scores[worst]
