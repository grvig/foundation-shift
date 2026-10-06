"""Linear probing: logistic regression on frozen embeddings.

Embeddings are L2-normalised and then standardised with statistics from the training
split only. Normalising first puts every backbone on the same footing (their raw vector
norms differ by orders of magnitude); standardising lets one regularisation grid suit all
of them and lets the solver converge in a sensible number of iterations.

Every value of C is fitted once and scored on every split. Selecting C is then a lookup,
so choosing on the out-of-distribution validation split and choosing on the
in-distribution one cost nothing extra and are reported side by side.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import metric_function

EVALUATION_SPLITS = ["id_val", "ood_val", "test"]


def l2_normalise(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return features / norms


class LinearProbe:
    def __init__(self, c, max_iter, seed):
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.scaler = None
        self.classifier = None

    def fit(self, features, labels):
        self.scaler = StandardScaler()
        prepared = self.scaler.fit_transform(l2_normalise(features))
        self.classifier = LogisticRegression(C=self.c, max_iter=self.max_iter,
                                             random_state=self.seed)
        self.classifier.fit(prepared, labels)
        return self

    def predict(self, features):
        if self.classifier is None:
            raise RuntimeError("fit the probe before predicting")
        return self.classifier.predict(self.scaler.transform(l2_normalise(features)))


def sweep(features, labels, splits, c_grid, metric_name, max_iter, seed, log=print):
    """Fit one probe per C on train; score each on every evaluation split.

    ``splits`` is an array of split names aligned with the rows of ``features``. Returns
    a list of row dicts and the fitted probes keyed by C.
    """
    metric = metric_function(metric_name)
    splits = np.asarray(splits)
    labels = np.asarray(labels)
    train = splits == "train"
    rows = []
    probes = {}
    for c in c_grid:
        probe = LinearProbe(c, max_iter, seed).fit(features[train], labels[train])
        probes[float(c)] = probe
        row = {"c": float(c)}
        for split in EVALUATION_SPLITS:
            mask = splits == split
            row[split] = metric(labels[mask], probe.predict(features[mask]))
        log("  C=" + format(c, "g") + "  id_val " + format(row["id_val"], ".4f")
            + "  ood_val " + format(row["ood_val"], ".4f"))
        rows.append(row)
    return rows, probes


def select(rows, selection_split):
    """The row with the best score on ``selection_split``; ties go to the smaller C."""
    best = None
    for row in sorted(rows, key=row_c):
        if best is None or row[selection_split] > best[selection_split]:
            best = row
    return best


def row_c(row):
    return row["c"]
