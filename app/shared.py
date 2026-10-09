"""Data loading shared by the app's pages. Everything shown comes from results/ and the
predictions saved under the data root; the app never runs a model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.data.datasets import dataset_dir
from src.data.datasets import load_metadata
from src.plots.figures import BACKBONE_LABELS

DATASET = "camelyon17"
CLASS_NAMES = ["normal", "tumour"]
METHOD_NAMES = {"linear_probe": "probe", "zero_shot": "zero-shot", "wise_ft": "WiSE-FT",
                "finetune_last2": "fine-tuned"}
SELECTION_NAMES = {"ood_val": "", "none": "", "id_val": ", C on familiar val"}


@st.cache_resource
def config():
    return load_config()


@st.cache_resource
def metadata():
    return load_metadata(config(), DATASET)


@st.cache_resource
def image_root():
    return dataset_dir(config(), DATASET)


def method_label(key):
    """Readable name for a predictions file such as clip_b16_linear_probe_ood_val."""
    for backbone, name in BACKBONE_LABELS.items():
        if not key.startswith(backbone + "_"):
            continue
        rest = key[len(backbone) + 1:]
        for method, method_name in METHOD_NAMES.items():
            if not rest.startswith(method + "_"):
                continue
            selection = rest[len(method) + 1:]
            return name.split(" (")[0] + ", " + method_name + SELECTION_NAMES.get(
                selection, ", " + selection)
    return key


@st.cache_resource
def predictions():
    """Every saved prediction file, keyed by its file name, in a stable display order."""
    directory = config().data_root() / "predictions" / DATASET
    saved = {}
    if not directory.exists():
        return saved
    for path in sorted(directory.glob("*.npy")):
        if "_seed" in path.stem:
            continue
        saved[path.stem] = np.load(path)
    return saved


def read_result(name):
    path = config().path("results_dir") / (DATASET + "_" + name + ".csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def percent(value):
    return format(100.0 * float(value), ".1f")
