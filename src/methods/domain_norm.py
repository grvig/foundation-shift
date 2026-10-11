"""Per-domain feature standardisation: the simplest use of unlabelled images from a new site.

A linear probe standardises features with statistics from the training hospitals only, so
a new hospital whose stain shifts every feature a little arrives off-centre. Here each
hospital's embeddings are instead centred and scaled with that hospital's own mean and
standard deviation: the training hospitals with theirs, the validation and test
hospitals with theirs. Only images are used from the new hospitals, never labels, so
this is an unsupervised adaptation that a deployment could do once it has collected a
batch of unlabelled patches. The report has to say that it is transductive: it assumes
access to the new site's images before predicting on them.

Memory: ResNet-50 embeddings are 456k x 2048 values. They are kept in float16, normalised
in place, and one domain at a time is promoted to float32 for the arithmetic, so the
peak is the stored array plus one hospital's rows.
"""

import numpy as np


def standardise_by_domain(features, domains):
    """L2-normalise each row, then standardise each domain with its own statistics.

    Works in place on ``features`` and also returns it. Feature dimensions with no
    spread inside a domain are left centred rather than divided by zero.
    """
    for domain in np.unique(domains):
        rows = np.flatnonzero(domains == domain)
        block = features[rows].astype(np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        block = block / norms
        mean = block.mean(axis=0)
        spread = block.std(axis=0)
        spread[spread == 0.0] = 1.0
        features[rows] = ((block - mean) / spread).astype(features.dtype)
    return features


class PreparedScaler:
    """For features that are already standardised: only converts to float32."""

    def fit(self, features):
        return self

    def transform(self, features):
        return np.asarray(features, dtype=np.float32)
