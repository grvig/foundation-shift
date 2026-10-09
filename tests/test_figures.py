import pytest
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


def test_keys_are_read_even_when_backbone_names_contain_underscores():
    from src.plots.figures import parse_key

    assert parse_key("clip_b16_linear_probe_ood_val") == ("clip_b16", "linear_probe")
    assert parse_key("resnet50_finetune_last2_none") == ("resnet50", "finetune_last2")
    with pytest.raises(ValueError):
        parse_key("vgg_linear_probe_none")


def test_the_forest_plot_has_a_panel_per_reference(tmp_path):
    from src.plots.figures import plot_paired

    paired = pd.DataFrame([
        ["resnet50_linear_probe_ood_val", "clip_b16", "linear_probe", "ood_val", 0.01, -0.02, 0.05],
        ["resnet50_linear_probe_ood_val", "clip_b16", "zero_shot", "none", -0.3, -0.4, -0.2],
        ["resnet50_finetune_last2_none", "dinov2_b14", "finetune_last2", "none", 0.04, 0.02, 0.06],
    ], columns=["reference", "backbone", "method", "selection", "difference", "low", "high"])
    plot_paired(paired, tmp_path / "paired")
    assert (tmp_path / "paired.png").stat().st_size > 1000


def test_both_figures_are_written_as_pdf_and_png(tmp_path):
    plot_main(MAIN, tmp_path / "main")
    fit = pd.Series({"slope": 1.0, "intercept": -0.3, "r": 0.9})
    plot_on_the_line(POINTS, fit, tmp_path / "line")
    for name in ["main", "line"]:
        assert (tmp_path / (name + ".pdf")).stat().st_size > 1000
        assert (tmp_path / (name + ".png")).stat().st_size > 1000
