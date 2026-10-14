"""The app's table helper: grouping, numeric alignment and escaping."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import style


class Capture:
    """Stands in for st.markdown and keeps what would have been drawn."""

    def __init__(self):
        self.blocks = []

    def __call__(self, text, unsafe_allow_html=False):
        self.blocks.append(text)


@pytest.fixture
def drawn(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(style.st, "markdown", capture)
    return capture


COLUMNS = [("backbone", "Backbone", "text"), ("score", "Score", "num strong")]


def test_a_group_label_prints_once_and_a_rule_marks_the_next_group(drawn):
    rows = [{"backbone": "CLIP", "score": "95.0"}, {"backbone": "CLIP", "score": "89.7"},
            {"backbone": "DINOv2", "score": "95.7"}]
    style.table(rows, COLUMNS, group="backbone")
    html = drawn.blocks[0]
    assert html.count(">CLIP<") == 1
    assert html.count("<tr class='group'>") == 1
    assert "<th class='num'>Score</th>" in html
    assert "<td class='num strong'>95.0</td>" in html


def test_cell_text_is_escaped(drawn):
    style.table([{"backbone": "<b>x</b>", "score": "1"}], COLUMNS)
    assert "&lt;b&gt;x&lt;/b&gt;" in drawn.blocks[0]


def test_a_caption_follows_the_table_when_given(drawn):
    style.table([{"backbone": "a", "score": "1"}], COLUMNS, caption="<b>Table 1.</b>")
    assert len(drawn.blocks) == 2
    assert "Table 1." in drawn.blocks[1]
    style.table([{"backbone": "a", "score": "1"}], COLUMNS)
    assert len(drawn.blocks) == 3
