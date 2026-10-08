"""run_main end to end, on fake embeddings for one backbone in a temporary data root."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_main
from src.config import DEFAULT_CONFIG_PATH
from src.config import DATA_ROOT_VARIABLE
from src.config import load_config
from src.data.datasets import load_metadata
from src.embeddings.store import FINAL
from src.embeddings.store import fingerprint
from src.embeddings.store import save_array
from src.embeddings.store import store_dir
from src.embeddings.store import write_manifest
from src.methods.zero_shot import TextHead
from src.models.backbones import backbone_settings


def make_metadata(directory, rng):
    lines = [",patient,node,x_coord,y_coord,tumor,slide,center,split"]
    for index in range(400):
        center = [0, 3, 4, 1, 2][index % 5]
        split = int(index % 7 == 0)
        tumor = int(rng.integers(0, 2))
        lines.append(str(index) + ",001,0," + str(index) + ",0," + str(tumor) + ","
                     + str(center * 10) + "," + str(center) + "," + str(split))
    directory.mkdir(parents=True)
    (directory / "metadata.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path / "data"))
    values = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    values["paths"]["results_dir"] = str(tmp_path / "results")
    values["paths"]["figures_dir"] = str(tmp_path / "figures")
    values["linear_probe"]["c_grid"] = [0.1, 1.0]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    make_metadata(tmp_path / "data" / "camelyon17_v1.0", rng)
    metadata = load_metadata(config, "camelyon17")
    paths = list(metadata["path"])
    # Embeddings that carry the label in their first coordinate, so the probe can learn it.
    features = rng.normal(size=(len(paths), 16))
    features[:, 0] = features[:, 0] + 4.0 * (2 * metadata["label"].to_numpy() - 1)
    for backbone in ["resnet50", "clip_b16"]:
        directory = store_dir(config.data_root(), "camelyon17", backbone)
        directory.mkdir(parents=True)
        save_array(directory / FINAL, features.astype(np.float16))
        write_manifest(directory, {"fingerprint": fingerprint(
            paths, backbone_settings(config, backbone)), "rows": len(paths),
            "dimension": 16, "shard_size": 1000, "complete": True})
    # A text head that points class 1 along the label axis, standing in for CLIP's.
    vectors = np.zeros((2, 16))
    vectors[0, 0] = -1.0
    vectors[1, 0] = 1.0
    monkeypatch.setattr(run_main, "load_text_head", fake_text_head(vectors))
    return config_path, tmp_path / "results"


def fake_text_head(vectors):
    def load(config, backbone, dataset, device):
        return TextHead(vectors, 100.0, 0.0)
    return load


def test_the_main_table_has_both_selection_rules(setup, monkeypatch):
    config_path, results = setup
    monkeypatch.setattr(sys, "argv", ["run_main.py", "camelyon17", "--config",
                                      str(config_path)])
    run_main.main()
    table = pd.read_csv(results / "camelyon17_main.csv")
    assert list(table["backbone"]) == ["resnet50"] * 2 + ["clip_b16"] * 5
    assert list(table["method"]) == ["linear_probe"] * 4 + ["zero_shot"] + ["wise_ft"] * 2
    assert list(table["selection"]) == ["ood_val", "id_val"] * 2 + ["none"] + ["ood_val",
                                                                            "id_val"]
    assert (table["test"] > 0.9).all()
    wise = table[table["method"] == "wise_ft"]
    assert wise["alpha"].between(0.0, 1.0).all()
    sweep = pd.read_csv(results / "camelyon17_wise_ft_sweep.csv")
    assert len(sweep) == 2 * 11
    assert (table["test_low"] <= table["test"]).all()
    assert (table["test"] <= table["test_high"]).all()
    saved = results.parent / "data" / "predictions" / "camelyon17"
    assert (saved / "resnet50_linear_probe_ood_val.npy").exists()
    domains = pd.read_csv(results / "camelyon17_per_domain.csv")
    id_rows = domains[(domains["split"] == "id_val") & (domains["selection"] == "ood_val")
                     & (domains["backbone"] == "resnet50")]
    assert sorted(id_rows["domain"]) == [0, 3, 4]
    test_rows = domains[domains["split"] == "test"]
    assert set(test_rows["domain"]) == {2}
    sweep = pd.read_csv(results / "camelyon17_probe_sweep.csv")
    assert list(sweep["c"]) == [0.1, 1.0] * 2


def test_backbones_without_embeddings_are_skipped(setup, monkeypatch, capsys):
    config_path, _ = setup
    monkeypatch.setattr(sys, "argv", ["run_main.py", "camelyon17", "--config",
                                      str(config_path)])
    run_main.main()
    assert "skipping siglip_b16" in capsys.readouterr().out
