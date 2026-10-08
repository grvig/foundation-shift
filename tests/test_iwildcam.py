import pytest

from src.data.iwildcam import class_names
from src.data.iwildcam import load_metadata

METADATA = """split,location_remapped,y,filename,sequence_remapped
train,0,0,a.jpg,1
id_val,0,1,b.jpg,2
val,7,1,c.jpg,3
test,9,2,d.jpg,4
id_test,0,0,e.jpg,5
"""


def test_wilds_splits_map_onto_the_project_vocabulary(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA)
    frame = load_metadata(tmp_path)
    assert list(frame["split"]) == ["train", "id_val", "ood_val", "test", "id_test"]
    assert list(frame["label"]) == [0, 1, 1, 2, 0]
    assert frame["path"][2] == "train/c.jpg"


def test_numeric_split_codes_are_understood(tmp_path):
    (tmp_path / "metadata.csv").write_text("split,location_remapped,y,filename\n"
                                           "0,0,0,a.jpg\n1,7,1,c.jpg\n3,0,1,b.jpg\n")
    frame = load_metadata(tmp_path)
    assert list(frame["split"]) == ["train", "ood_val", "id_val"]


def test_unseen_traps_are_never_in_training(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA)
    frame = load_metadata(tmp_path)
    training = set(frame[frame["split"].isin(["train", "id_val", "id_test"])]
                   ["location_remapped"])
    held_out = set(frame[frame["split"].isin(["ood_val", "test"])]["location_remapped"])
    assert training.isdisjoint(held_out)


def test_an_unknown_split_fails_loudly(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA.replace("id_test", "holdout"))
    with pytest.raises(ValueError):
        load_metadata(tmp_path)


def test_class_names_come_from_categories_with_visible_gaps(tmp_path):
    (tmp_path / "categories.csv").write_text("y,category_id,name\n0,0,empty\n2,5,"
                                             "tayassu pecari\n")
    assert class_names(tmp_path, [0, 1, 2]) == ["empty", "species 1", "tayassu pecari"]


def test_the_config_uses_the_official_metric_and_camera_traps():
    from src.config import load_config

    settings = load_config().dataset("iwildcam")
    assert settings["metric"] == "macro_f1"
    assert settings["domain_column"] == "location_remapped"
    assert settings["cluster_column"] == "location_remapped"
