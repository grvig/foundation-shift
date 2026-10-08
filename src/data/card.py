"""The data card: what each split contains.

Two things on it matter for reading every later result. First, how unequal the domains
are in size, since a pooled score is dominated by the largest. Second, the label balance
per split: if it changes between training and test domains, part of any score drop is a
change in what is being predicted rather than in how the images look, and the report has
to say so.

Datasets with a handful of domains (Camelyon17's five hospitals) get one row per split and
domain. Datasets with hundreds (iWildCam's camera traps) get one row per split, with the
number of domains in it, because a row per trap would be unreadable.
"""

import pandas as pd

SPLIT_ORDER = ["train", "id_val", "ood_val", "test", "id_test"]
MAX_DOMAIN_ROWS = 10


def describe(group, split, domain, domain_column, cluster_column):
    counts = group["label"].value_counts()
    row = {"split": split, "domain": domain, "images": len(group),
           "domains": int(group[domain_column].nunique()),
           "clusters": int(group[cluster_column].nunique()),
           "classes": int(group["label"].nunique()),
           "majority_share": float(counts.iloc[0] / len(group))}
    if set(group["label"].unique()) <= {0, 1}:
        row["positive_share"] = float(group["label"].mean())
    return row


def summarise(metadata, domain_column, cluster_column):
    rows = []
    for split in SPLIT_ORDER:
        part = metadata[metadata["split"] == split]
        if len(part) == 0:
            continue
        domains = sorted(part[domain_column].unique())
        if len(domains) > MAX_DOMAIN_ROWS:
            rows.append(describe(part, split, "all", domain_column, cluster_column))
            continue
        for domain in domains:
            group = part[part[domain_column] == domain]
            rows.append(describe(group, split, str(domain), domain_column, cluster_column))
    return pd.DataFrame(rows)
