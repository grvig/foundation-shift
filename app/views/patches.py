"""Real tissue patches with two methods' answers side by side."""

import numpy as np
import streamlit as st
from PIL import Image

import shared
import style

FILTERS = ["any patch", "A right, B wrong", "A wrong, B right", "both wrong", "both right"]
SPLITS = {"test hospital (2)": "test", "validation hospital (1)": "ood_val",
          "familiar hospitals (0, 3, 4)": "id_val"}
GRID = 12
COLUMNS = 6

style.header("Camelyon17 · hospital shift",
             "Patches",
             "Each square is 96 by 96 pixels of stained lymph node tissue, labelled by "
             "whether its centre holds tumour. The models never saw the test hospital. "
             "Choose two methods and look at the patches where they part ways.")

saved = shared.predictions()
if len(saved) == 0:
    st.warning("No saved predictions yet. Run scripts/run_main.py camelyon17 first.")
    st.stop()

keys = list(saved.keys())


def position_of(key):
    if key in keys:
        return keys.index(key)
    return 0


controls = st.columns([3, 3, 2, 2])
method_a = controls[0].selectbox("Method A", keys,
                                 index=position_of("dinov2_b14_finetune_last2_none"),
                                 format_func=shared.method_label)
method_b = controls[1].selectbox("Method B", keys,
                                 index=position_of("resnet50_linear_probe_ood_val"),
                                 format_func=shared.method_label)
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

style.note(format(len(candidates), ",") + " of " + format(int(in_split.sum()), ",")
           + " patches match &nbsp;·&nbsp; A correct on "
           + shared.percent(right_a[in_split].mean()) + "% &nbsp;·&nbsp; B correct on "
           + shared.percent(right_b[in_split].mean()) + "%")

if "draw" not in st.session_state:
    st.session_state["draw"] = 0
if st.button("Draw another twelve"):
    st.session_state["draw"] = st.session_state["draw"] + 1

if len(candidates) == 0:
    st.info("No patch matches this filter.")
else:
    rng = np.random.default_rng(st.session_state["draw"])
    chosen = rng.choice(candidates, size=min(GRID, len(candidates)), replace=False)
    columns = st.columns(COLUMNS, gap="small")
    for position, row in enumerate(chosen):
        with columns[position % COLUMNS]:
            path = shared.image_root() / metadata["path"][int(row)]
            with Image.open(path) as image:
                st.image(image.convert("RGB").resize((192, 192), Image.NEAREST),
                         width="stretch")
            lines = ["<b>" + shared.CLASS_NAMES[int(labels[row])] + "</b> · slide "
                     + str(int(metadata["slide"][row]))]
            for name, key, right in [("A", method_a, right_a), ("B", method_b, right_b)]:
                css = "wrong"
                if right[row]:
                    css = "right"
                lines.append(name + " <span class='" + css + "'>"
                             + shared.CLASS_NAMES[int(saved[key][row])] + "</span>")
            st.markdown("<div class='patch'>" + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True)

style.section("Slides",
              "Accuracy slide by slide",
              "Patches from one slide share its stain and its patient, so a method tends "
              "to win or lose a whole slide at once. That is why every interval in this "
              "project resamples slides, not patches.")
rows = []
slides = metadata["slide"].to_numpy()
for slide in np.unique(slides[in_split]):
    mask = in_split & (slides == slide)
    rows.append({"slide": str(int(slide)), "patches": format(int(mask.sum()), ","),
                 "tumour": shared.percent(labels[mask].mean()),
                 "a": shared.percent(right_a[mask].mean()),
                 "b": shared.percent(right_b[mask].mean())})
style.table(rows, [("slide", "Slide", "text"), ("patches", "Patches", "num"),
                   ("tumour", "Tumour %", "num"), ("a", "A correct %", "num"),
                   ("b", "B correct %", "num")],
            "<b>A</b> " + shared.method_label(method_a) + " &nbsp; <b>B</b> "
            + shared.method_label(method_b))
