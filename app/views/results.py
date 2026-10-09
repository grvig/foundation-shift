"""The results tables and figures, read straight from results/ and figures/."""

import pandas as pd
import streamlit as st

import shared

st.title("Results on Camelyon17")
st.write("Accuracy in percent. Familiar hospitals are the three seen in training; the "
         "test hospital was never seen. The interval resamples whole slides.")

main = shared.read_result("main")
if main is None:
    st.warning("No results yet. Run scripts/run_all.py camelyon17.")
    st.stop()
finetune = shared.read_result("finetune")
if finetune is not None:
    main = pd.concat([main, finetune], ignore_index=True)

rows = []
for _, row in main.iterrows():
    key = row["backbone"] + "_" + row["method"] + "_" + row["selection"]
    rows.append({"method": shared.method_label(key),
                 "familiar hospitals": shared.percent(row["id_val"]),
                 "validation hospital": shared.percent(row["ood_val"]),
                 "test hospital": shared.percent(row["test"]),
                 "test interval": shared.percent(row["test_low"]) + " to "
                 + shared.percent(row["test_high"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

figures = shared.config().path("figures_dir")
columns = st.columns(2)
for column, name, caption in [
    (columns[0], "camelyon17_main.png", "Familiar versus test hospital, with the test "
     "interval. Diamonds are zero-shot."),
    (columns[1], "camelyon17_on_the_line.png", "In-distribution accuracy barely predicts "
     "test-hospital accuracy here."),
]:
    path = figures / name
    if path.exists():
        column.image(str(path), use_container_width=True)
        column.caption(caption)

paired = shared.read_result("paired")
if paired is not None:
    st.subheader("Is the difference real?")
    st.write("Test accuracy minus the reference's, with a 95% interval from resampling "
             "the same slides for both. A difference counts only when the interval "
             "excludes zero.")
    rows = []
    for _, row in paired.iterrows():
        key = row["backbone"] + "_" + row["method"] + "_" + row["selection"]
        verdict = "no clear difference"
        if bool(row["significant"]) and row["difference"] > 0:
            verdict = "better"
        if bool(row["significant"]) and row["difference"] < 0:
            verdict = "worse"
        rows.append({"against": shared.method_label(row["reference"]),
                     "method": shared.method_label(key),
                     "difference": format(100.0 * row["difference"], "+.1f"),
                     "interval": format(100.0 * row["low"], "+.1f") + " to "
                     + format(100.0 * row["high"], "+.1f"),
                     "verdict": verdict})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    forest = figures / "camelyon17_paired.png"
    if forest.exists():
        st.image(str(forest), width=700)
        st.caption("Zero-shot rows are left out of the plot; their 30 to 40 point deficits "
                   "would squeeze every other interval. They are in the table above.")

seeds = shared.read_result("finetune_seed_summary")
if seeds is not None:
    st.subheader("Fine-tuning across training runs")
    st.write("The same fine-tuning repeated with different seeds: a different training "
             "subset, batch order and starting head. The slide interval cannot see this "
             "kind of variation.")
    rows = []
    for _, row in seeds.iterrows():
        spread = "-"
        if row["seeds"] > 1:
            spread = format(100.0 * row["test_sd"], ".1f")
        rows.append({"backbone": shared.method_label(row["backbone"] + "_finetune_last2_none"),
                     "runs": int(row["seeds"]), "mean test": shared.percent(row["test_mean"]),
                     "sd": spread, "lowest": shared.percent(row["test_min"]),
                     "highest": shared.percent(row["test_max"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

blocks = shared.read_result("blocks_summary")
if blocks is not None:
    st.subheader("How much of the model to fine-tune")
    st.write("The same fine-tuning with one, two or four final blocks trainable, one run "
             "each. Compare the gains with the seed spread above before reading much into "
             "them.")
    rows = []
    for _, row in blocks.iterrows():
        rows.append({"backbone": shared.BACKBONE_LABELS[row["backbone"]],
                     "blocks": int(row["blocks"]),
                     "familiar hospitals": shared.percent(row["id_val"]),
                     "test hospital": shared.percent(row["test"]),
                     "test interval": shared.percent(row["test_low"]) + " to "
                     + shared.percent(row["test_high"])})
    columns = st.columns([3, 2])
    columns[0].dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    path = figures / "camelyon17_blocks.png"
    if path.exists():
        columns[1].image(str(path), use_container_width=True)
