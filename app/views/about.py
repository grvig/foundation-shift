"""What the project tests and how, in plain words."""

import streamlit as st

st.title("About")
st.markdown("""
**The question.** Large pretrained image models are often called robust. This project
checks whether they keep working when they are deployed somewhere they have never seen,
using the WILDS benchmark's real shifts instead of synthetic corruptions.

**The data.** Camelyon17-WILDS: 455,954 tissue patches from five hospitals, each labelled
tumour or normal. Models train on three hospitals. Hospital 1 is used for validation and
hospital 2 for the final test; neither appears in training.

**The models.** An ImageNet-trained ResNet-50, the pre-foundation-model baseline, and
three foundation models: CLIP and SigLIP, trained on image-text pairs from the web, and
DINOv2, trained on images alone.

**The methods.**

- *Zero-shot*: describe each class in words and pick the closest description. No
  training at all.
- *Linear probe*: train only a final linear layer on the frozen model's features.
- *WiSE-FT*: blend the zero-shot classifier with the probe.
- *Fine-tuning*: also retrain the model's last two blocks.

**The evidence.** Every score comes with a 95% interval from resampling whole slides,
because thousands of patches from one slide are not thousands of independent tests.
Two methods are compared on the same resampled slides, and a difference only counts when
its interval excludes zero.

The code, data checks and every number are in the repository README.
""")
