import pytest

from src.data.camelyon17 import load_metadata
from src.data.card import summarise

METADATA = """,patient,node,x_coord,y_coord,tumor,slide,center,split
0,004,4,1,1,1,0,0,0
1,004,4,2,1,0,0,0,0
2,005,1,3,1,1,1,0,1
3,010,1,4,1,0,5,1,0
4,051,2,5,1,1,30,2,0
5,052,2,6,1,1,31,2,1
"""


def test_the_card_counts_by_split_and_hospital(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA)
    card = summarise(load_metadata(tmp_path, val_center=1, test_center=2))
    assert list(card["split"]) == ["train", "id_val", "ood_val", "test"]
    assert list(card["patches"]) == [2, 1, 1, 2]
    train = card.iloc[0]
    assert train["tumour_share"] == pytest.approx(0.5)
    test = card.iloc[3]
    assert test["slides"] == 2
    assert test["patients"] == 2
