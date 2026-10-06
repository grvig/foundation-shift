"""The data card: what each split contains, hospital by hospital.

Two things on it matter for reading every later result. First, how unequal the
hospitals are in size, since a pooled in-distribution score is dominated by the largest.
Second, the tumour share per hospital: if it differs between training and test hospitals,
part of any accuracy drop is a change in label balance rather than in image appearance,
and the report has to say so.
"""

import pandas as pd

from src.data.camelyon17 import SPLIT_ORDER


def summarise(metadata):
    rows = []
    for split in SPLIT_ORDER:
        part = metadata[metadata["split"] == split]
        for center in sorted(part["center"].unique()):
            group = part[part["center"] == center]
            rows.append({
                "split": split,
                "center": int(center),
                "patches": len(group),
                "slides": int(group["slide"].nunique()),
                "patients": int(group["patient"].nunique()),
                "tumour_share": float(group["label"].mean()),
            })
    return pd.DataFrame(rows)
