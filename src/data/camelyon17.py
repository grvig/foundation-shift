"""Camelyon17-WILDS: 96x96 tissue patches labelled tumour or normal, from five hospitals.

The archive contains ``metadata.csv`` and a ``patches/`` folder. Each row of the metadata
is one patch: the patient and lymph node it came from, its position on the slide, the
slide id, the hospital (``center``) and the label (``tumor``).

Splits follow the official WILDS definition exactly, so our numbers are comparable with
the published leaderboard:

``train``
    Hospitals 0, 3 and 4, rows whose ``split`` column is 0.
``id_val``
    The same three hospitals, rows whose ``split`` column is 1. This is the
    in-distribution evaluation set; Camelyon17 has no separate in-distribution test split.
``ood_val``
    Every patch from hospital 1. Used for model selection.
``test``
    Every patch from hospital 2. Touched only for final numbers.

The ``split`` column in the file only distinguishes train from id_val. The hospital
assignment overrides it, which is how the WILDS package itself builds the split.

Patient ids are zero-padded strings in the file ("004"), and the padding is part of every
image path, so they are read as strings rather than integers.
"""

import pandas as pd

METADATA_FILE = "metadata.csv"
FILE_SPLIT_NAMES = {0: "train", 1: "id_val"}
SPLIT_ORDER = ["train", "id_val", "ood_val", "test"]


def load_metadata(dataset_dir, val_center, test_center):
    """The metadata table with a ``split`` name and an image ``path`` for every patch."""
    path = dataset_dir / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError("no " + METADATA_FILE + " in " + str(dataset_dir)
                                + "; run scripts/prepare_data.py first")
    frame = pd.read_csv(path, index_col=0, dtype={"patient": str})
    # Row position is the image id the mirror and the embeddings are keyed on, so the
    # file's own index column must be exactly 0, 1, 2, ... for that to be safe.
    if list(frame.index) != list(range(len(frame))):
        raise ValueError(METADATA_FILE + " rows are not numbered 0 to n-1 in order")
    frame = frame.reset_index(drop=True)
    frame["split"] = assign_splits(frame, val_center, test_center)
    frame["path"] = image_paths(frame)
    frame["label"] = frame["tumor"].astype(int)
    return frame


def assign_splits(frame, val_center, test_center):
    codes = frame["split"].astype(int)
    unknown = sorted(set(codes.unique()) - set(FILE_SPLIT_NAMES.keys()))
    if len(unknown) > 0:
        raise ValueError("unexpected split code in metadata: " + str(unknown[0]))
    names = codes.map(FILE_SPLIT_NAMES).to_numpy(dtype=object)
    centers = frame["center"].astype(int).to_numpy()
    names[centers == int(val_center)] = "ood_val"
    names[centers == int(test_center)] = "test"
    return list(names)


def image_paths(frame):
    """Relative path of every patch, in the layout the WILDS archive uses.

    Built with whole-column string operations: a row-by-row loop took most of a minute
    on 456k rows, and every script and the app load this table.
    """
    stem = ("patient_" + frame["patient"].astype(str) + "_node_"
            + frame["node"].astype(int).astype(str))
    name = ("patch_" + stem + "_x_" + frame["x_coord"].astype(int).astype(str) + "_y_"
            + frame["y_coord"].astype(int).astype(str) + ".png")
    return list("patches/" + stem + "/" + name)


MIRROR_FIELDS = [("patient", "patient"), ("node", "node"), ("x_coord", "x_coord"),
                 ("y_coord", "y_coord"), ("slide", "slide"), ("center", "center"),
                 ("tumor", "label")]


def mirror_mismatches(official, mirror):
    """Fields where a mirror row disagrees with the official row it claims to be.

    The mirror's ``image_id`` is the row number in the official metadata. Patient ids
    are compared as numbers, since the official file pads them ("004") and the mirror
    does not.
    """
    wrong = []
    for official_field, mirror_field in MIRROR_FIELDS:
        if int(official[official_field]) != int(mirror[mirror_field]):
            wrong.append(official_field)
    return wrong


def split_frame(frame, name):
    if name not in SPLIT_ORDER:
        raise ValueError("unknown split: " + name)
    return frame[frame["split"] == name].reset_index(drop=True)
