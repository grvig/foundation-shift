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
