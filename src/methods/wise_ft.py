"""WiSE-FT for linear classifiers: blend the zero-shot head with the trained one.

WiSE-FT (weight-space ensembling) interpolates between a zero-shot model and its
fine-tuned version, and was reported to keep most of the fine-tuned in-distribution
accuracy while recovering much of the zero-shot model's robustness to shift. Its cheapest
form touches only the final linear layer: both classifiers act on the same frozen image
embedding, so interpolating their weights by a mixing coefficient alpha is the same as
interpolating their logits,

    logits(alpha) = (1 - alpha) * zero_shot_logits + alpha * probe_logits,

with alpha = 0 the zero-shot classifier and alpha = 1 the linear probe. Everything works
from cached embeddings, so the whole alpha sweep costs seconds.

The two heads here are trained independently rather than one starting from the other,
so their logits need not share a scale. The zero-shot logits use the model's own learned
temperature; the sweep over alpha absorbs what remains of the mismatch, and alpha is
chosen on a validation split like any other hyperparameter.
"""

import numpy as np

from src.evaluation.metrics import metric_function
from src.methods.linear_probe import EVALUATION_SPLITS


def blend(zero_shot_logits, probe_logits, alpha):
    return (1.0 - float(alpha)) * zero_shot_logits + float(alpha) * probe_logits


def alpha_sweep(zero_shot_logits, probe_logits, labels, splits, alphas, metric_name):
    """Score every alpha on every evaluation split; one row dict per alpha."""
    metric = metric_function(metric_name)
    labels = np.asarray(labels)
    splits = np.asarray(splits)
    rows = []
    for alpha in alphas:
        predictions = np.argmax(blend(zero_shot_logits, probe_logits, alpha), axis=1)
        row = {"alpha": float(alpha)}
        for split in EVALUATION_SPLITS:
            mask = splits == split
            row[split] = metric(labels[mask], predictions[mask])
        rows.append(row)
    return rows


def select_alpha(rows, selection_split):
    """The best alpha on ``selection_split``; ties go to the smaller alpha, which keeps
    more of the zero-shot classifier."""
    best = None
    for row in sorted(rows, key=row_alpha):
        if best is None or row[selection_split] > best[selection_split]:
            best = row
    return best


def row_alpha(row):
    return row["alpha"]
