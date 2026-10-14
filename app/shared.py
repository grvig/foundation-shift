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
                "finetune_last1": "fine-tuned, 1 block",
                "finetune_last2": "fine-tuned",
                "finetune_last4": "fine-tuned, 4 blocks",
                "finetune_last2_aug": "fine-tuned with stain augmentation",
                "probe_domain_norm": "probe, per-hospital standardisation"}
SELECTION_NAMES = {"ood_val": "", "none": "", "id_val": ", C on familiar val"}
CHANGE_NAMES = {"finetune_last2": "fine-tuning, vs probe",
                "finetune_last2_aug": "stain augmentation, vs plain fine-tuning",
                "probe_domain_norm": "per-hospital standardisation, vs plain probe"}


@st.cache_resource(show_spinner=False)
def config():
    return load_config()


@st.cache_resource(show_spinner="Loading 455,954 patch records")
def metadata():
    return load_metadata(config(), DATASET)


@st.cache_resource(show_spinner=False)
def image_root():
    return dataset_dir(config(), DATASET)


BACKBONE_ORDER = ["resnet50", "clip_b16", "siglip_b16", "dinov2_b14"]
METHOD_ORDER = ["linear_probe", "zero_shot", "wise_ft", "finetune_last2"]


def short_backbone(backbone):
    return BACKBONE_LABELS.get(backbone, backbone).split(" (")[0]


def method_only(method, selection):
    """A method's name without its backbone, e.g. 'probe, C on familiar val'."""
    return METHOD_NAMES.get(method, method) + SELECTION_NAMES.get(selection, ", " + selection)


def in_report_order(table, then=None):
    """Rows sorted by backbone, then method, the order every table in the app uses.

    Tables with a selection column put the WILDS rule (ood_val) before the familiar one;
    ``then`` names one more column to sort by last, such as the block count.
    """
    ordered = table.copy()
    ordered["_b"] = ordered["backbone"].map(backbone_position)
    ordered["_m"] = ordered["method"].map(method_position)
    keys = ["_b", "_m"]
    ascending = [True, True]
    if "selection" in ordered.columns:
        keys.append("selection")
        ascending.append(False)
    if then is not None:
        keys.append(then)
        ascending.append(True)
    ordered = ordered.sort_values(keys, ascending=ascending)
    return ordered.drop(columns=["_b", "_m"]).reset_index(drop=True)


def backbone_position(backbone):
    if backbone in BACKBONE_ORDER:
        return BACKBONE_ORDER.index(backbone)
    return len(BACKBONE_ORDER)


def method_position(method):
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def method_label(key):
    """Readable name for a predictions file such as clip_b16_linear_probe_ood_val."""
    for backbone, name in BACKBONE_LABELS.items():
        if not key.startswith(backbone + "_"):
            continue
        rest = key[len(backbone) + 1:]
        # Longest match first: finetune_last2_aug also starts with finetune_last2.
        found = None
        for method in METHOD_NAMES:
            if rest.startswith(method + "_"):
                if found is None or len(method) > len(found):
                    found = method
        if found is None:
            continue
        selection = rest[len(found) + 1:]
        return (name.split(" (")[0] + ", " + METHOD_NAMES[found]
                + SELECTION_NAMES.get(selection, ", " + selection))
    return key


@st.cache_resource(show_spinner="Loading saved predictions")
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
