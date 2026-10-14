"""What the project tests and how, written like a short methods section."""

import streamlit as st

import shared
import style

style.header("Camelyon17 · hospital shift",
             "Method",
             "Large pretrained image models are often called robust. This project checks "
             "whether that holds when a model trained at some hospitals is used at one it "
             "has never seen, on a real shift rather than an artificial one.")

style.section("1", "Data",
              "Camelyon17-WILDS: tissue patches from lymph node sections, each labelled "
              "by whether its centre holds tumour. Hospitals 0, 3 and 4 train the models; "
              "hospital 1 validates and hospital 2 tests, and neither is seen in training. "
              "Labels, hospitals and splits come from the official WILDS metadata, and "
              "every image was checked against it.")
card = shared.read_result("data_card")
if card is not None:
    names = {"train": "train", "id_val": "familiar validation", "ood_val": "new, validation",
             "test": "new, test"}
    rows = []
    for _, row in card.iterrows():
        rows.append({"split": names.get(row["split"], row["split"]),
                     "hospital": str(row["domain"]),
                     "patches": format(int(row["images"]), ","),
                     "slides": str(int(row["clusters"])),
                     "tumour": shared.percent(row["positive_share"])})
    style.table(rows, [("split", "Split", "text"), ("hospital", "Hospital", "num"),
                       ("patches", "Patches", "num"), ("slides", "Slides", "num"),
                       ("tumour", "Tumour %", "num")],
                "<b>Table 1.</b> Every hospital is close to half tumour, so a drop at a new "
                "hospital comes from how its images look, not from a different mix of "
                "cases.", group="split")

style.section("2", "Models",
              "One image model from before the foundation-model era and three after it, "
              "all run at 224 pixels with the whole patch resized the same way for each.")
style.table([
    {"model": "ResNet-50", "trained on": "ImageNet, 1.3M labelled photos", "type": "supervised"},
    {"model": "CLIP ViT-B/16", "trained on": "400M web image-text pairs", "type": "image and text"},
    {"model": "SigLIP ViT-B/16", "trained on": "web image-text pairs, sigmoid loss",
     "type": "image and text"},
    {"model": "DINOv2 ViT-B/14", "trained on": "142M curated images, no labels",
     "type": "self-supervised"},
], [("model", "Model", "strong"), ("trained on", "Pretrained on", "text"),
    ("type", "Kind", "quiet")])

style.section("3", "Methods",
              "From cheapest to most adaptation. Only per-hospital standardisation uses "
              "anything from the new hospital, and then only its images, never its labels.")
style.table([
    {"method": "Zero-shot", "how": "describe each class in words, pick the closest description",
     "trains": "nothing"},
    {"method": "Linear probe", "how": "logistic regression on the frozen model's features",
     "trains": "one linear layer"},
    {"method": "WiSE-FT", "how": "blend the zero-shot and probe classifiers",
     "trains": "one linear layer"},
    {"method": "Per-hospital standardisation",
     "how": "probe on features re-centred with each hospital's own statistics",
     "trains": "one linear layer"},
    {"method": "Fine-tuning", "how": "retrain the last two blocks with a new head",
     "trains": "about 9 to 21M parameters"},
    {"method": "Stain augmentation", "how": "fine-tuning with random colour, flips and turns",
     "trains": "about 9 to 21M parameters"},
], [("method", "Method", "strong"), ("how", "How it works", "text"),
    ("trains", "What trains", "quiet")])

style.section("4", "Evidence",
              "The test hospital has 85,054 patches but only ten slides, and patches from "
              "one slide share its stain and its patient. So every 95 percent interval "
              "resamples whole slides rather than patches. Two methods are compared on the "
              "same resampled slides, and a difference counts only when its interval stays "
              "on one side of zero. Fine-tuning was repeated with three seeds to measure "
              "how much a result moves between training runs, which the slide interval "
              "cannot see.")

style.section("5", "Reproducing",
              "Everything in this app is read from the repository's results and figures "
              "folders; the app never runs a model. <code>python scripts/run_all.py "
              "camelyon17</code> regenerates them from the cached embeddings.")
