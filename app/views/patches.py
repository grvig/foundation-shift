"""Real tissue patches with two methods' answers side by side."""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import shared

FILTERS = ["any patch", "A right, B wrong", "A wrong, B right", "both wrong", "both right"]
SPLITS = {"test hospital (2)": "test", "validation hospital (1)": "ood_val",
          "familiar hospitals (0, 3, 4)": "id_val"}
GRID = 12

st.title("Browse patches")
st.write("Each patch is 96 by 96 pixels of stained lymph node tissue. The label says "
         "whether its centre contains tumour. Pick two methods and see where they "
         "disagree.")

saved = shared.predictions()
if len(saved) == 0:
    st.warning("No saved predictions yet. Run scripts/run_main.py camelyon17 first.")
    st.stop()

keys = list(saved.keys())


def position_of(key):
    if key in keys:
        return keys.index(key)
    return 0


default_a = position_of("dinov2_b14_finetune_last2_none")
default_b = position_of("resnet50_linear_probe_ood_val")

controls = st.columns([3, 3, 2, 2])
method_a = controls[0].selectbox("Method A", keys, index=default_a, format_func=shared.method_label)
method_b = controls[1].selectbox("Method B", keys, index=default_b, format_func=shared.method_label)
where = controls[2].selectbox("Hospital", list(SPLITS.keys()))
which = controls[3].selectbox("Show", FILTERS, index=1)

metadata = shared.metadata()
labels = metadata["label"].to_numpy()
in_split = metadata["split"].to_numpy() == SPLITS[where]
right_a = saved[method_a] == labels
right_b = saved[method_b] == labels
masks = {"any patch": in_split, "A right, B wrong": in_split & right_a & ~right_b,
         "A wrong, B right": in_split & ~right_a & right_b,
         "both wrong": in_split & ~right_a & ~right_b,
         "both right": in_split & right_a & right_b}
candidates = np.flatnonzero(masks[which])

total = int(in_split.sum())
st.markdown('<p class="small">' + format(len(candidates), ",") + " of " + format(total, ",")
            + " patches in this hospital match. Method A is right on "
            + shared.percent(right_a[in_split].mean()) + "% of them, method B on "
            + shared.percent(right_b[in_split].mean()) + "%.</p>", unsafe_allow_html=True)

if "draw" not in st.session_state:
    st.session_state["draw"] = 0
if st.button("Show different patches"):
    st.session_state["draw"] = st.session_state["draw"] + 1

if len(candidates) == 0:
    st.info("No patch matches this filter.")
else:
    rng = np.random.default_rng(st.session_state["draw"])
    chosen = rng.choice(candidates, size=min(GRID, len(candidates)), replace=False)
    columns = st.columns(6)
    for position, row in enumerate(chosen):
        with columns[position % 6]:
            path = shared.image_root() / metadata["path"][int(row)]
            with Image.open(path) as image:
                st.image(image.convert("RGB").resize((192, 192), Image.NEAREST),
                         use_container_width=True)
            truth = shared.CLASS_NAMES[int(labels[row])]
            lines = ["true: <b>" + truth + "</b>"]
            for name, key, right in [("A", method_a, right_a), ("B", method_b, right_b)]:
                css = "wrong"
                if right[row]:
                    css = "right"
                lines.append(name + ": <span class='" + css + "'>"
                             + shared.CLASS_NAMES[int(saved[key][row])] + "</span>")
            lines.append("<span class='small'>slide " + str(int(metadata["slide"][row]))
                         + "</span>")
            st.markdown("<div class='patch-caption'>" + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True)

st.subheader("Accuracy slide by slide")
st.write("Patches from one slide share its staining and its patient, so methods tend to "
         "win or lose whole slides at a time. This is why the confidence intervals "
         "resample slides rather than patches.")
rows = []
slides = metadata["slide"].to_numpy()
for slide in np.unique(slides[in_split]):
    mask = in_split & (slides == slide)
    rows.append({"slide": int(slide), "patches": int(mask.sum()),
                 "tumour %": shared.percent(labels[mask].mean()),
                 "A %": shared.percent(right_a[mask].mean()),
                 "B %": shared.percent(right_b[mask].mean())})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
