"""iWildCam-WILDS: camera-trap photos labelled with one of 182 species.

The archive has ``metadata.csv`` and the images under ``train/``. Each row is one photo:
its file name, the species label ``y``, the camera trap it came from
(``location_remapped``) and the WILDS split.

The official split is by camera trap. Training traps supply ``train`` and two
in-distribution sets (``id_val``, ``id_test``); traps never seen in training supply the
out-of-distribution ``val`` and ``test``. The names are mapped onto this project's split
vocabulary so every script handles both datasets the same way:

=============  ===========
WILDS split    this project
=============  ===========
train          train
id_val         id_val
val            ood_val
test           test
id_test        id_test
=============  ===========

``id_test`` is kept but not used by the main tables, which report ``id_val`` as the
in-distribution number for both datasets.
"""

import pandas as pd

METADATA_FILE = "metadata.csv"
CATEGORIES_FILE = "categories.csv"
SPLIT_NAMES = {"train": "train", "id_val": "id_val", "val": "ood_val", "test": "test",
               "id_test": "id_test"}
SPLIT_CODES = {0: "train", 1: "val", 2: "test", 3: "id_val", 4: "id_test"}
IMAGE_FOLDER = "train"


def wilds_split_name(value):
    """The WILDS split name, whether the file stores it as a word or as its code."""
    if isinstance(value, str) and not value.isdigit():
        return value
    code = int(value)
    if code not in SPLIT_CODES:
        raise ValueError("unexpected split code in metadata: " + str(value))
    return SPLIT_CODES[code]


def load_metadata(dataset_dir):
    path = dataset_dir / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError("no " + METADATA_FILE + " in " + str(dataset_dir)
                                + "; run scripts/prepare_data.py first")
    frame = pd.read_csv(path)
    names = []
    for value in frame["split"]:
        wilds_name = wilds_split_name(value)
        if wilds_name not in SPLIT_NAMES:
            raise ValueError("unexpected split in metadata: " + str(value))
        names.append(SPLIT_NAMES[wilds_name])
    frame["split"] = names
    frame["label"] = frame["y"].astype(int)
    paths = []
    for name in frame["filename"]:
        paths.append(IMAGE_FOLDER + "/" + str(name))
    frame["path"] = paths
    return frame


def class_names(dataset_dir, labels):
    """Species names in label order, for zero-shot prompts.

    Read from categories.csv when the archive has it. A label without a name gets a
    placeholder, which keeps prompts aligned with labels and makes the gap visible.
    """
    path = dataset_dir / CATEGORIES_FILE
    names = {}
    if path.exists():
        categories = pd.read_csv(path)
        for _, row in categories.iterrows():
            names[int(row["y"])] = str(row["name"])
    ordered = []
    for label in range(int(max(labels)) + 1):
        ordered.append(names.get(label, "species " + str(label)))
    return ordered
