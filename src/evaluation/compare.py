"""Every method against one reference, on the same resampled slides.

Two methods' test scores each come with an interval, but overlapping intervals do not mean
"no difference": both scores move together when a hard slide is drawn, so most of each
interval's width is shared. The paired bootstrap resamples slides once per step and scores
both methods on that draw, so only the difference between them varies. A difference is
called significant when its 95% interval excludes zero.
"""

import pandas as pd

from src.evaluation.bootstrap import paired_cluster_bootstrap


def method_key(row):
    return row["backbone"] + "_" + row["method"] + "_" + row["selection"]


def is_reference(row, reference):
    for field in ["backbone", "method", "selection"]:
        if row[field] != reference[field]:
            return False
    return True


def compare_to_reference(table, load_predictions, labels, clusters, metric_name,
                         resamples, make_rng, reference):
    """One row per non-reference method: its test score minus the reference's, with CI.

    ``labels`` and ``clusters`` are the test split only, and ``load_predictions(key)``
    must return predictions for those same rows. ``make_rng`` gives a fresh generator
    per comparison so each one is reproducible on its own.
    """
    reference_rows = []
    for _, row in table.iterrows():
        if is_reference(row, reference):
            reference_rows.append(row)
    if len(reference_rows) != 1:
        raise ValueError("the reference method must appear exactly once in the table")
    reference_predictions = load_predictions(method_key(reference_rows[0]))

    rows = []
    for _, row in table.iterrows():
        if is_reference(row, reference):
            continue
        difference, low, high = paired_cluster_bootstrap(
            labels, load_predictions(method_key(row)), reference_predictions, clusters,
            metric_name, resamples, make_rng())
        significant = low > 0.0 or high < 0.0
        rows.append({"backbone": row["backbone"], "method": row["method"],
                     "selection": row["selection"], "test": row["test"],
                     "difference": difference, "low": low, "high": high,
                     "significant": significant})
    return pd.DataFrame(rows)
