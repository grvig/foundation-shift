"""The results, laid out like the results section of a paper."""

import pandas as pd
import streamlit as st

import shared
import style

style.header("Camelyon17 · hospital shift",
             "Results",
             "Accuracy in percent. The familiar hospitals are the three the models trained "
             "on; the new ones were never seen. Intervals are 95 percent and come from "
             "resampling whole tissue slides, because the test hospital has only ten.")

main = shared.read_result("main")
if main is None:
    st.warning("No results yet. Run scripts/run_all.py camelyon17.")
    st.stop()
finetune = shared.read_result("finetune")
if finetune is not None:
    main = pd.concat([main, finetune], ignore_index=True)
figures = shared.config().path("figures_dir")

style.section("1", "Accuracy at a new hospital",
              "Every backbone with every method. The bold column is the hospital none "
              "of them saw during training.")
rows = []
for _, row in shared.in_report_order(main).iterrows():
    rows.append({"backbone": shared.short_backbone(row["backbone"]),
                 "method": shared.method_only(row["method"], row["selection"]),
                 "familiar": shared.percent(row["id_val"]),
                 "validation": shared.percent(row["ood_val"]),
                 "test": shared.percent(row["test"]),
                 "interval": shared.percent(row["test_low"]) + "–"
                 + shared.percent(row["test_high"])})
style.table(rows, [("backbone", "Backbone", "text"), ("method", "Method", "text"),
                   ("familiar", "Familiar", "num"), ("validation", "New, val", "num"),
                   ("test", "New, test", "num strong"), ("interval", "Test 95% CI", "num")],
            "<b>Table 1.</b> Familiar is the in-distribution validation split of hospitals "
            "0, 3 and 4; new is hospital 1 (validation) and hospital 2 (test).",
            group="backbone")
style.figure(figures / "camelyon17_main.png",
             "<b>Figure 1.</b> Hollow circles: familiar hospitals. Filled markers: the "
             "test hospital with its slide interval, circles for probes and squares for "
             "fine-tuned models. Diamonds: zero-shot.")

paired = shared.read_result("paired")
if paired is not None:
    style.section("2", "Is the difference real?",
                  "Each method's test accuracy minus a reference's, with both scored on "
                  "the same resampled slides. A difference counts only when its interval "
                  "stays on one side of zero.")
    references = list(dict.fromkeys(paired["reference"]))
    reference = st.radio("Compared against", references, horizontal=True,
                         format_func=shared.method_label)
    chosen = shared.in_report_order(paired[paired["reference"] == reference])
    rows = []
    for _, row in chosen.iterrows():
        verdict = "no clear difference"
        if bool(row["significant"]) and row["difference"] > 0:
            verdict = "better"
        if bool(row["significant"]) and row["difference"] < 0:
            verdict = "worse"
        rows.append({"backbone": shared.short_backbone(row["backbone"]),
                     "method": shared.method_only(row["method"], row["selection"]),
                     "difference": format(100.0 * row["difference"], "+.1f"),
                     "interval": format(100.0 * row["low"], "+.1f") + " to "
                     + format(100.0 * row["high"], "+.1f"),
                     "verdict": verdict})
    style.table(rows, [("backbone", "Backbone", "text"), ("method", "Method", "text"),
                       ("difference", "Difference", "num strong"),
                       ("interval", "95% CI", "num"), ("verdict", "Verdict", "quiet")],
                "<b>Table 2.</b> Percentage points against "
                + shared.method_label(reference) + ".", group="backbone")
    style.figure(figures / "camelyon17_paired.png",
                 "<b>Figure 2.</b> The same differences for all three references. "
                 "Zero-shot and WiSE-FT rows are in the table but not the plot: zero-shot "
                 "deficits would squash the axis, and WiSE-FT sits on its probe.", width=640)

seeds = shared.read_result("finetune_seed_summary")
if seeds is not None:
    style.section("3", "Repeated training runs",
                  "Fine-tuning repeated with three seeds: a different training subset, "
                  "batch order and starting head each time. The slide interval cannot see "
                  "this kind of variation, so it is measured separately.")
    seeds = seeds.copy()
    seeds["method"] = "finetune_last2"
    rows = []
    for _, row in shared.in_report_order(seeds).iterrows():
        spread = "–"
        if row["seeds"] > 1:
            spread = format(100.0 * row["test_sd"], ".1f")
        rows.append({"backbone": shared.short_backbone(row["backbone"]),
                     "runs": str(int(row["seeds"])),
                     "mean": shared.percent(row["test_mean"]), "sd": spread,
                     "range": shared.percent(row["test_min"]) + "–"
                     + shared.percent(row["test_max"])})
    style.table(rows, [("backbone", "Backbone", "text"), ("runs", "Runs", "num"),
                       ("mean", "Mean test", "num strong"), ("sd", "SD", "num"),
                       ("range", "Lowest–highest", "num")],
                "<b>Table 3.</b> Test-hospital accuracy of the fine-tuned models across "
                "seeds.")

blocks = shared.read_result("blocks_summary")
if blocks is not None:
    style.section("4", "How much of the model to fine-tune",
                  "One, two or four final blocks trainable, one run each. Read the gaps "
                  "against the seed spread in Table 3.")
    blocks = blocks.copy()
    blocks["method"] = "finetune_last2"
    rows = []
    for _, row in shared.in_report_order(blocks, then="blocks").iterrows():
        rows.append({"backbone": shared.short_backbone(row["backbone"]),
                     "blocks": str(int(row["blocks"])),
                     "familiar": shared.percent(row["id_val"]),
                     "test": shared.percent(row["test"]),
                     "interval": shared.percent(row["test_low"]) + "–"
                     + shared.percent(row["test_high"])})
    style.table(rows, [("backbone", "Backbone", "text"), ("blocks", "Blocks", "num"),
                       ("familiar", "Familiar", "num"), ("test", "New, test", "num strong"),
                       ("interval", "Test 95% CI", "num")],
                "<b>Table 4.</b> Accuracy by number of fine-tuned blocks.", group="backbone")
    style.figure(shared.config().path("figures_dir") / "camelyon17_blocks.png",
                 "<b>Figure 3.</b> Squares: test hospital with its interval. Circles: "
                 "familiar hospitals.", width=640)

within = shared.read_result("paired_within")
if within is not None and len(within) > 0:
    style.section("5", "Changes within one backbone",
                  "Each change against the same backbone without it, on the same resampled "
                  "slides: fine-tuning against the probe, stain augmentation against plain "
                  "fine-tuning, and per-hospital standardisation against the plain probe.")
    rows = []
    for _, row in within.iterrows():
        verdict = "no clear difference"
        if bool(row["significant"]) and row["difference"] > 0:
            verdict = "better"
        if bool(row["significant"]) and row["difference"] < 0:
            verdict = "worse"
        rows.append({"change": shared.CHANGE_NAMES.get(row["method"], row["method"]),
                     "backbone": shared.short_backbone(row["backbone"]),
                     "difference": format(100.0 * row["difference"], "+.1f"),
                     "interval": format(100.0 * row["low"], "+.1f") + " to "
                     + format(100.0 * row["high"], "+.1f"),
                     "verdict": verdict})
    style.table(rows, [("change", "Change", "text"), ("backbone", "Backbone", "text"),
                       ("difference", "Difference", "num strong"),
                       ("interval", "95% CI", "num"), ("verdict", "Verdict", "quiet")],
                "<b>Table 5.</b> Percentage points of test accuracy.", group="change")
    style.figure(shared.config().path("figures_dir") / "camelyon17_within.png",
                 "<b>Figure 4.</b> The differences in Table 5.", width=640)

fit = shared.read_result("on_the_line_fit")
line_text = ("If it did, the probe points would fall along a rising line, as they do on "
             "many natural-image benchmarks.")
if fit is not None:
    line_text = (line_text + " Across the " + str(int(fit["points"][0])) + " probe "
                 "settings the correlation in probit space is r = "
                 + format(float(fit["r"][0]), ".2f") + ".")
style.section("6", "Does familiar accuracy predict new accuracy?", line_text)
style.figure(shared.config().path("figures_dir") / "camelyon17_on_the_line.png",
             "<b>Figure 5.</b> Familiar against test-hospital accuracy on probit axes. "
             "Solid line: fitted to the probes. Dotted: no drop at all.", width=560)
