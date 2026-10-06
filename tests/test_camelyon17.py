"""Split assignment and paths for Camelyon17, on a small hand-written metadata file."""

import pytest

from src.data.camelyon17 import load_metadata
from src.data.camelyon17 import split_frame

# Same shape as the real file: an unnamed index column first, zero-padded patient ids.
METADATA = """,patient,node,x_coord,y_coord,tumor,slide,center,split
0,004,4,3328,21792,1,0,0,0
1,004,4,3200,22272,0,0,0,1
2,010,1,1000,2000,0,5,1,0
3,010,1,1096,2000,1,5,1,1
4,051,2,640,960,1,30,2,0
5,061,0,512,512,0,35,3,0
6,072,3,96,192,0,40,4,1
"""


@pytest.fixture
def metadata(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA)
    return load_metadata(tmp_path, val_center=1, test_center=2)


def test_hospital_overrides_the_split_column(metadata):
    assert list(metadata["split"]) == ["train", "id_val", "ood_val", "ood_val", "test",
                                       "train", "id_val"]


def test_no_held_out_hospital_reaches_training(metadata):
    training = metadata[metadata["split"].isin(["train", "id_val"])]
    assert set(training["center"]) == {0, 3, 4}


def test_patient_padding_survives_into_the_path(metadata):
    expected = "patches/patient_004_node_4/patch_patient_004_node_4_x_3328_y_21792.png"
    assert metadata["path"][0] == expected


def test_labels_are_integers(metadata):
    assert list(metadata["label"]) == [1, 0, 0, 1, 1, 0, 0]


def test_split_frame_selects_one_split(metadata):
    test = split_frame(metadata, "test")
    assert len(test) == 1
    assert test["patient"][0] == "051"
    with pytest.raises(ValueError):
        split_frame(metadata, "validation")


def test_an_unknown_split_code_fails_loudly(tmp_path):
    text = METADATA.replace("0,004,4,3328,21792,1,0,0,0", "0,004,4,3328,21792,1,0,0,7")
    (tmp_path / "metadata.csv").write_text(text)
    with pytest.raises(ValueError):
        load_metadata(tmp_path, val_center=1, test_center=2)


def test_a_missing_file_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_metadata(tmp_path, val_center=1, test_center=2)


def test_the_config_matches_the_official_split():
    from src.config import load_config

    settings = load_config().dataset("camelyon17")
    assert settings["val_center"] == 1
    assert settings["test_center"] == 2
    assert len(settings["mirror_revision"]) == 40
