import pandas as pd
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


def test_few_domains_get_a_row_each(tmp_path):
    (tmp_path / "metadata.csv").write_text(METADATA)
    card = summarise(load_metadata(tmp_path, val_center=1, test_center=2), "center",
                     "slide")
    assert list(card["split"]) == ["train", "id_val", "ood_val", "test"]
    assert list(card["images"]) == [2, 1, 1, 2]
    assert card.iloc[0]["positive_share"] == pytest.approx(0.5)
    assert card.iloc[3]["clusters"] == 2


def test_many_domains_are_summarised_per_split():
    frame = pd.DataFrame({"split": ["train"] * 12 + ["test"] * 3,
                          "location": list(range(12)) + [20, 21, 21],
                          "label": [0] * 9 + [1, 2, 2] + [5, 5, 7]})
    card = summarise(frame, "location", "location")
    train = card.iloc[0]
    assert train["domain"] == "all"
    assert train["domains"] == 12
    assert train["classes"] == 3
    assert train["majority_share"] == pytest.approx(0.75)
    assert "positive_share" not in card.columns or pd.isna(train["positive_share"])
    test = card[card["split"] == "test"]
    assert list(test["domain"]) == ["20", "21"]
