"""Report figures, drawn from the CSVs in results/ and nothing else.

Each backbone keeps one colour in every figure, assigned in a fixed order so that a
figure with fewer backbones never repaints the ones it shows. The four colours were
checked for colour-blind separation; two of them are low-contrast on white, so every
point is also named in text (axis labels or direct labels) and colour is never the only
way to tell backbones apart.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.on_the_line import predicted_ood
from src.evaluation.on_the_line import probit

BACKBONE_COLOURS = {"resnet50": "#2a78d6", "clip_b16": "#eb6834",
                    "siglip_b16": "#1baf7a", "dinov2_b14": "#eda100"}
BACKBONE_LABELS = {"resnet50": "ResNet-50 (ImageNet)", "clip_b16": "CLIP ViT-B/16",
                   "siglip_b16": "SigLIP ViT-B/16", "dinov2_b14": "DINOv2 ViT-B/14"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(MUTED)
    axes.spines["bottom"].set_color(MUTED)
    axes.tick_params(colors=MUTED, labelcolor=INK)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def ordered_backbones(names):
    ordered = []
    for name in BACKBONE_COLOURS:
        if name in list(names):
            ordered.append(name)
    return ordered


def plot_main(table, path):
    """Per backbone: familiar-hospital accuracy, test-hospital accuracy with its slide
    interval, and zero-shot test accuracy where the backbone has a text encoder."""
    probes = table[(table["method"] == "linear_probe") & (table["selection"] == "ood_val")]
    backbones = ordered_backbones(probes["backbone"])
    figure, axes = plt.subplots(figsize=(7.0, 0.6 * len(backbones) + 1.4))
    for position, backbone in enumerate(backbones):
        y = len(backbones) - 1 - position
        colour = BACKBONE_COLOURS[backbone]
        row = probes[probes["backbone"] == backbone].iloc[0]
        axes.plot([row["test"], row["id_val"]], [y, y], color=GRID, linewidth=2, zorder=1)
        axes.errorbar(row["test"], y, xerr=[[row["test"] - row["test_low"]],
                      [row["test_high"] - row["test"]]], fmt="o", color=colour,
                      markersize=8, linewidth=2, capsize=0, zorder=3,
                      markeredgecolor="white", markeredgewidth=1.5)
        axes.plot(row["id_val"], y, "o", markersize=8, markerfacecolor="white",
                  markeredgecolor=colour, markeredgewidth=2, zorder=3)
        zero_shot = table[(table["backbone"] == backbone) & (table["method"] == "zero_shot")]
        if len(zero_shot) > 0:
            axes.plot(zero_shot.iloc[0]["test"], y, "D", markersize=7, color=colour,
                      markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    axes.set_yticks(range(len(backbones)))
    labels = []
    for backbone in reversed(backbones):
        labels.append(BACKBONE_LABELS[backbone])
    axes.set_yticklabels(labels)
    axes.set_xlabel("accuracy")
    style(axes)
    axes.grid(False, axis="y")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                   markeredgecolor=MUTED, markeredgewidth=2, markersize=8),
        plt.Line2D([], [], marker="o", linestyle="-", color=MUTED, markersize=8),
        plt.Line2D([], [], marker="D", linestyle="", color=MUTED, markersize=7),
    ]
    axes.legend(handles, ["familiar hospitals", "new hospital, 95% interval",
                          "zero-shot, new hospital"], loc="lower center",
                bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=8,
                handletextpad=0.3, columnspacing=1.0)
    figure.tight_layout()
    save(figure, path)


def plot_on_the_line(points, fit, path):
    """In-distribution versus test accuracy on probit axes, with the fitted line."""
    figure, axes = plt.subplots(figsize=(5.2, 4.6))
    low = min(points["id_val"].min(), points["test"].min()) - 0.02
    high = min(max(points["id_val"].max(), points["test"].max()) + 0.01, 0.999)
    grid = np.linspace(low, high, 100)
    axes.plot(probit(grid), probit(grid), linestyle=":", color=MUTED, linewidth=1)
    # The line is drawn only across the probe points it was fitted on; extending it to
    # the zero-shot points would show an extrapolation as if it were evidence.
    probe_ids = points[points["method"] == "linear_probe"]["id_val"]
    fitted = np.linspace(probe_ids.min(), probe_ids.max(), 50)
    axes.plot(probit(fitted), probit(predicted_ood(fitted, fit["slope"], fit["intercept"])),
              color=MUTED, linewidth=2)
    for backbone in ordered_backbones(points["backbone"]):
        colour = BACKBONE_COLOURS[backbone]
        probes = points[(points["backbone"] == backbone)
                        & (points["method"] == "linear_probe")]
        axes.plot(probit(probes["id_val"]), probit(probes["test"]), "o", color=colour,
                  markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                  label=BACKBONE_LABELS[backbone])
        zero_shot = points[(points["backbone"] == backbone)
                           & (points["method"] == "zero_shot")]
        if len(zero_shot) > 0:
            axes.plot(probit(zero_shot["id_val"]), probit(zero_shot["test"]), "D",
                      color=colour, markersize=8, markeredgecolor="white",
                      markeredgewidth=1.5)
    ticks = []
    for value in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        if low <= value <= high:
            ticks.append(value)
    axes.set_xticks(probit(ticks))
    axes.set_xticklabels([format(t, "g") for t in ticks])
    axes.set_yticks(probit(ticks))
    axes.set_yticklabels([format(t, "g") for t in ticks])
    axes.set_xlabel("accuracy, familiar hospitals (probit scale)")
    axes.set_ylabel("accuracy, new hospital (probit scale)")
    axes.set_title("solid: fitted line   dotted: no drop   diamonds: zero-shot",
                   fontsize=8, color=MUTED)
    style(axes)
    axes.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    save(figure, path)


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(figure)
