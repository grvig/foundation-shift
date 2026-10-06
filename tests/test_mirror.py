"""Unpacking the mirror, on a tiny parquet file built like the real one."""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.data.camelyon17 import load_metadata
from src.data.mirror import write_images

METADATA = """,patient,node,x_coord,y_coord,tumor,slide,center,split
0,004,4,3328,21792,1,0,0,0
1,010,1,1000,2000,0,5,1,0
2,051,2,640,960,1,30,2,0
"""


def png(colour):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def mirror_row(image_id, patient, node, x, y, slide, center, label):
    return {"image": {"bytes": png((image_id * 80, 0, 0)), "path": str(image_id) + ".png"},
            "image_id": image_id, "patient": patient, "node": node, "x_coord": x,
            "y_coord": y, "slide": slide, "center": center, "label": label}


GOOD_ROWS = [mirror_row(2, 51, 2, 640, 960, 30, 2, 1),
             mirror_row(0, 4, 4, 3328, 21792, 0, 0, 1),
             mirror_row(1, 10, 1, 1000, 2000, 5, 1, 0)]


def setup(tmp_path, rows):
    (tmp_path / "metadata.csv").write_text(METADATA)
    metadata = load_metadata(tmp_path, val_center=1, test_center=2)
    parquet = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    return metadata, parquet


def quiet(message):
    pass


def test_images_land_at_their_official_paths(tmp_path):
    metadata, parquet = setup(tmp_path, GOOD_ROWS)
    assert write_images([parquet], metadata, tmp_path, log=quiet) == 3
    path = tmp_path / "patches/patient_051_node_2/patch_patient_051_node_2_x_640_y_960.png"
    assert Image.open(path).getpixel((0, 0)) == (160, 0, 0)
    assert len(list(tmp_path.rglob("*.tmp"))) == 0


def test_a_rerun_skips_finished_images(tmp_path):
    metadata, parquet = setup(tmp_path, GOOD_ROWS)
    write_images([parquet], metadata, tmp_path, log=quiet)
    assert write_images([parquet], metadata, tmp_path, log=quiet) == 0


def test_a_wrong_label_is_refused(tmp_path):
    rows = list(GOOD_ROWS)
    rows[0] = mirror_row(2, 51, 2, 640, 960, 30, 2, 0)
    metadata, parquet = setup(tmp_path, rows)
    with pytest.raises(ValueError, match="disagrees on tumor"):
        write_images([parquet], metadata, tmp_path, log=quiet)


def test_a_wrong_hospital_is_refused(tmp_path):
    rows = list(GOOD_ROWS)
    rows[1] = mirror_row(0, 4, 4, 3328, 21792, 0, 3, 1)
    metadata, parquet = setup(tmp_path, rows)
    with pytest.raises(ValueError, match="center"):
        write_images([parquet], metadata, tmp_path, log=quiet)


def test_a_missing_row_is_refused(tmp_path):
    metadata, parquet = setup(tmp_path, GOOD_ROWS[:2])
    with pytest.raises(ValueError, match="1 official rows have no image"):
        write_images([parquet], metadata, tmp_path, log=quiet)


def test_a_duplicated_row_is_refused(tmp_path):
    metadata, parquet = setup(tmp_path, GOOD_ROWS + [GOOD_ROWS[0]])
    with pytest.raises(ValueError, match="appears twice"):
        write_images([parquet], metadata, tmp_path, log=quiet)


def test_metadata_rows_out_of_order_are_refused(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA.replace("\n1,010", "\n7,010"))
    with pytest.raises(ValueError, match="numbered"):
        load_metadata(tmp_path, val_center=1, test_center=2)
