"""Headless runs of every app page. Skipped when the dataset is not on this machine."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from src.config import load_config
from src.data.datasets import dataset_dir


def data_present():
    try:
        dataset_dir(load_config(), "camelyon17")
    except FileNotFoundError:
        return False
    return True


needs_data = pytest.mark.skipif(not data_present(), reason="Camelyon17 is not downloaded")


def test_method_labels_are_readable():
    import shared

    assert shared.method_label("dinov2_b14_finetune_last2_none") == "DINOv2 ViT-B/14, fine-tuned"
    assert shared.method_label("resnet50_linear_probe_id_val") == (
        "ResNet-50, probe, C on familiar val")
    assert shared.method_label("unknown_key") == "unknown_key"


@needs_data
@pytest.mark.parametrize("page", ["patches.py", "results.py", "about.py"])
def test_every_page_runs_without_an_exception(page):
    app = AppTest.from_file(str(APP / "views" / page), default_timeout=120)
    app.run()
    assert len(app.exception) == 0


@needs_data
def test_the_patch_filter_can_be_changed():
    app = AppTest.from_file(str(APP / "views" / "patches.py"), default_timeout=120)
    app.run()
    app.selectbox[3].select("both wrong").run()
    assert len(app.exception) == 0
    app.button[0].click().run()
    assert len(app.exception) == 0
