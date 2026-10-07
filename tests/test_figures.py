import pandas as pd

from src.plots.figures import BACKBONE_COLOURS
from src.plots.figures import ordered_backbones
from src.plots.figures import plot_main
from src.plots.figures import plot_on_the_line

MAIN = pd.DataFrame([
    ["resnet50", "linear_probe", "ood_val", 0.97, 0.90, 0.89, 0.82, 0.92],
    ["resnet50", "linear_probe", "id_val", 0.97, 0.90, 0.89, 0.83, 0.93],
    ["clip_b16", "linear_probe", "ood_val", 0.98, 0.93, 0.92, 0.88, 0.95],
    ["clip_b16", "zero_shot", "none", 0.60, 0.55, 0.52, 0.45, 0.60],
], columns=["backbone", "method", "selection", "id_val", "ood_val", "test", "test_low",
            "test_high"])

POINTS = pd.DataFrame([
    ["resnet50", "linear_probe", 0.96, 0.88],
    ["resnet50", "linear_probe", 0.97, 0.89],
    ["clip_b16", "linear_probe", 0.98, 0.92],
    ["clip_b16", "zero_shot", 0.60, 0.52],
], columns=["backbone", "method", "id_val", "test"])


def test_colours_follow_the_backbone_not_its_rank():
    assert ordered_backbones(["dinov2_b14", "resnet50"]) == ["resnet50", "dinov2_b14"]
    assert len(set(BACKBONE_COLOURS.values())) == len(BACKBONE_COLOURS)


def test_both_figures_are_written_as_pdf_and_png(tmp_path):
    plot_main(MAIN, tmp_path / "main")
    fit = pd.Series({"slope": 1.0, "intercept": -0.3, "r": 0.9})
    plot_on_the_line(POINTS, fit, tmp_path / "line")
    for name in ["main", "line"]:
        assert (tmp_path / (name + ".pdf")).stat().st_size > 1000
        assert (tmp_path / (name + ".png")).stat().st_size > 1000
