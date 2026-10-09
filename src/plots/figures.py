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
METHOD_SHORT = {"linear_probe": "probe", "zero_shot": "zero-shot", "wise_ft": "WiSE-FT",
                "finetune_last2": "fine-tuned"}
FINETUNE_METHOD = "finetune_last2"
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
    interval for the probe and (when present) the fine-tuned model, and zero-shot test
    accuracy where the backbone has a text encoder."""
    probes = table[(table["method"] == "linear_probe") & (table["selection"] == "ood_val")]
    tuned = table[table["method"] == FINETUNE_METHOD]
    backbones = ordered_backbones(probes["backbone"])
    figure, axes = plt.subplots(figsize=(7.0, 0.75 * len(backbones) + 1.6))
    for position, backbone in enumerate(backbones):
        y = len(backbones) - 1 - position
        colour = BACKBONE_COLOURS[backbone]
        row = probes[probes["backbone"] == backbone].iloc[0]
        probe_y = y
        if len(tuned) > 0:
            probe_y = y + 0.15
        axes.plot([row["test"], row["id_val"]], [probe_y, probe_y], color=GRID, linewidth=2,
                  zorder=1)
        draw_interval(axes, row, probe_y, "o", colour)
        axes.plot(row["id_val"], probe_y, "o", markersize=8, markerfacecolor="white",
                  markeredgecolor=colour, markeredgewidth=2, zorder=3)
        tuned_row = tuned[tuned["backbone"] == backbone]
        if len(tuned_row) > 0:
            draw_interval(axes, tuned_row.iloc[0], y - 0.15, "s", colour)
        zero_shot = table[(table["backbone"] == backbone) & (table["method"] == "zero_shot")]
        if len(zero_shot) > 0:
            axes.plot(zero_shot.iloc[0]["test"], probe_y, "D", markersize=7, color=colour,
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
    names = ["probe, familiar hospitals", "probe, new hospital (95% interval)",
             "zero-shot, new hospital"]
    if len(tuned) > 0:
        handles.append(plt.Line2D([], [], marker="s", linestyle="-", color=MUTED,
                                  markersize=7))
        names.append("fine-tuned, new hospital (95% interval)")
    axes.legend(handles, names, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2,
                frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
    figure.tight_layout()
    save(figure, path)


def draw_interval(axes, row, y, marker, colour):
    axes.errorbar(row["test"], y, xerr=[[row["test"] - row["test_low"]],
                  [row["test_high"] - row["test"]]], fmt=marker, color=colour,
                  markersize=8, linewidth=2, capsize=0, zorder=3,
                  markeredgecolor="white", markeredgewidth=1.5)


def short_label(backbone, method):
    return BACKBONE_LABELS[backbone].split(" (")[0] + ", " + METHOD_SHORT.get(method, method)


def parse_key(key):
    """(backbone, method) from a key such as clip_b16_linear_probe_ood_val."""
    for backbone in BACKBONE_LABELS:
        if not key.startswith(backbone + "_"):
            continue
        rest = key[len(backbone) + 1:]
        for method in METHOD_SHORT:
            if rest.startswith(method + "_"):
                return backbone, method
    raise ValueError("cannot read a backbone and method from " + key)


def plot_paired(paired, path):
    """Forest plot: each method's test accuracy minus its reference's, with the paired
    slide interval. One panel per reference. Zero-shot rows are left out because their
    50-point deficits would squeeze every other interval into a sliver; they are in the
    CSV."""
    paired = paired[(paired["method"] != "zero_shot") & (paired["selection"] != "id_val")]
    references = list(dict.fromkeys(paired["reference"]))
    heights = []
    for reference in references:
        heights.append(max(len(paired[paired["reference"] == reference]), 1))
    figure, panels = plt.subplots(len(references), 1, squeeze=False,
                                  figsize=(6.4, 0.32 * sum(heights) + 1.2 * len(references)),
                                  gridspec_kw={"height_ratios": heights})
    for panel, reference in zip(panels[:, 0], references):
        rows = paired[paired["reference"] == reference].reset_index(drop=True)
        labels = []
        for position, row in rows.iterrows():
            y = len(rows) - 1 - position
            colour = BACKBONE_COLOURS.get(row["backbone"], MUTED)
            marker = "o"
            if row["method"] == FINETUNE_METHOD:
                marker = "s"
            panel.errorbar(100 * row["difference"], y,
                           xerr=[[100 * (row["difference"] - row["low"])],
                                 [100 * (row["high"] - row["difference"])]],
                           fmt=marker, color=colour, markersize=7, linewidth=2, capsize=0,
                           markeredgecolor="white", markeredgewidth=1.2)
            labels.append(short_label(row["backbone"], row["method"]))
        panel.axvline(0, color=INK, linewidth=1)
        panel.set_yticks(range(len(rows)))
        panel.set_yticklabels(list(reversed(labels)), fontsize=8)
        reference_backbone, reference_method = parse_key(reference)
        panel.set_title("against " + short_label(reference_backbone, reference_method),
                        fontsize=9, color=MUTED, loc="left")
        style(panel)
        panel.grid(False, axis="y")
    panels[-1, 0].set_xlabel("test accuracy difference, percentage points")
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
