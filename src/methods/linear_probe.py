"""Linear probing: logistic regression on frozen embeddings.

Embeddings are L2-normalised and then standardised with statistics from the training
split only. Normalising first puts every backbone on the same footing (their raw vector
norms differ by orders of magnitude); standardising lets one regularisation grid suit all
of them and lets the solver converge in a sensible number of iterations.

Every value of C is fitted once and scored on every split. Selecting C is then a lookup,
so choosing on the out-of-distribution validation split and choosing on the
in-distribution one cost nothing extra and are reported side by side.

**Memory.** ResNet-50 embeddings for Camelyon17 are 456k x 2048 floats, 3.7 GB, on a
16 GB laptop. The sweep therefore prepares the training rows once and shares that copy
across every C (the preparation is identical for each, so this changes no result), and
prediction works through the rows in chunks rather than copying all of them at once.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import metric_function

EVALUATION_SPLITS = ["id_val", "ood_val", "test"]
CHUNK_ROWS = 50000


def l2_normalise(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return features / norms


class FeatureScaler:
    """L2 normalisation followed by standardisation fitted on the training rows."""

    def __init__(self):
        self.scaler = None

    def fit(self, features):
        self.scaler = StandardScaler()
        self.scaler.fit(l2_normalise(features))
        return self

    def transform(self, features):
        prepared = l2_normalise(features)
        self.scaler.transform(prepared, copy=False)
        return prepared.astype(np.float32, copy=False)


class LinearProbe:
    def __init__(self, c, max_iter, seed, scaler=None):
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.scaler = scaler
        self.classifier = None

    def fit(self, features, labels):
        """Fit on raw features, with a scaler fitted on them."""
        self.scaler = FeatureScaler().fit(features)
        return self.fit_prepared(self.scaler.transform(features), labels)

    def fit_prepared(self, prepared, labels):
        """Fit on rows already passed through ``self.scaler``."""
        self.classifier = LogisticRegression(C=self.c, max_iter=self.max_iter,
                                             random_state=self.seed)
        self.classifier.fit(prepared, labels)
        return self

    def decision(self, features):
        if self.classifier is None:
            raise RuntimeError("fit the probe before predicting")
        parts = []
        for start in range(0, len(features), CHUNK_ROWS):
            chunk = self.scaler.transform(features[start:start + CHUNK_ROWS])
            parts.append(self.classifier.decision_function(chunk))
        return np.concatenate(parts, axis=0)

    def predict(self, features):
        logits = self.logits(features)
        return self.classifier.classes_[np.argmax(logits, axis=1)]

    def logits(self, features):
        """One column per class. A binary model has a single decision value d, which is
        the same classifier as logits (0, d)."""
        decision = self.decision(features)
        if decision.ndim == 1:
            return np.stack([np.zeros_like(decision), decision], axis=1)
        return decision


def sweep(features, labels, splits, c_grid, metric_name, max_iter, seed, log=print):
    """Fit one probe per C on train; score each on every evaluation split.

    ``splits`` is an array of split names aligned with the rows of ``features``. Returns
    a list of row dicts and the fitted probes keyed by C.
    """
    metric = metric_function(metric_name)
    splits = np.asarray(splits)
    labels = np.asarray(labels)
    train = splits == "train"
    scaler = FeatureScaler().fit(features[train])
    prepared = scaler.transform(features[train])
    rows = []
    probes = {}
    for c in c_grid:
        probe = LinearProbe(c, max_iter, seed, scaler).fit_prepared(prepared, labels[train])
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
