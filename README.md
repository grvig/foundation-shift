# foundation-shift

Testing whether large pretrained vision models (CLIP, SigLIP, DINOv2) stay accurate when
deployed somewhere new - unseen camera traps and an unseen hospital - using the
[WILDS](https://wilds.stanford.edu/) benchmark.

## Question

Pretrained vision models are often described as robust. WILDS measures robustness on real
deployment shifts rather than synthetic corruptions:

| Dataset | Task | What shifts at test time | Official metric |
|---|---|---|---|
| Camelyon17 | tumour detection in tissue patches | the hospital | accuracy |
| iWildCam | species classification, 182 classes | the camera trap | macro-F1 |

We compare an ImageNet ResNet-50 against CLIP, SigLIP and DINOv2 under four adaptation
methods (zero-shot, linear probe, fine-tuning, weight-space ensembling) and report every
result twice: on familiar data (in-distribution) and on the unseen domain
(out-of-distribution).

## Results: Camelyon17

Accuracy in percent. Familiar hospitals are 0, 3 and 4 (the in-distribution validation
split); the new hospitals are 1 (validation) and 2 (test), neither seen in training. The
interval resamples whole tissue slides, because patches from one slide are not
independent evidence. Every hospital is about 50 percent tumour, so a drop between
columns reflects images that look different, not a different mix of cases.

<!-- results:camelyon17 -->
| Backbone | Method | Familiar hospitals | New hospital (val) | New hospital (test) | Test 95% interval |
|---|---|---|---|---|---|
| ResNet-50 (ImageNet) | linear probe (C on new-hospital val) | 97.0 | 90.5 | **88.8** | 82.4 - 92.4 |
| ResNet-50 (ImageNet) | linear probe (C on familiar val) | 97.0 | 90.3 | **89.5** | 83.3 - 92.8 |
| CLIP ViT-B/16 | linear probe (C on new-hospital val) | 95.9 | 88.5 | **89.7** | 85.6 - 92.8 |
| CLIP ViT-B/16 | linear probe (C on familiar val) | 95.9 | 88.5 | **89.2** | 85.0 - 92.5 |
| CLIP ViT-B/16 | zero-shot | 58.4 | 49.7 | **60.1** | 50.2 - 80.1 |
| CLIP ViT-B/16 | WiSE-FT (C on new-hospital val) | 96.0 | 88.6 | **89.7** | 85.7 - 93.0 |
| CLIP ViT-B/16 | WiSE-FT (C on familiar val) | 96.0 | 88.5 | **89.3** | 85.0 - 92.7 |
| SigLIP ViT-B/16 | linear probe (C on new-hospital val) | 96.6 | 91.8 | **87.4** | 79.5 - 91.9 |
| SigLIP ViT-B/16 | linear probe (C on familiar val) | 96.7 | 91.8 | **87.8** | 80.3 - 92.1 |
| SigLIP ViT-B/16 | zero-shot | 50.5 | 50.0 | **50.0** | 30.5 - 84.9 |
| SigLIP ViT-B/16 | WiSE-FT (C on new-hospital val) | 96.6 | 92.0 | **88.4** | 81.3 - 92.5 |
| SigLIP ViT-B/16 | WiSE-FT (C on familiar val) | 96.7 | 91.8 | **87.8** | 80.3 - 92.1 |
| DINOv2 ViT-B/14 | linear probe (C on new-hospital val) | 97.1 | 92.1 | **91.7** | 89.3 - 95.2 |
| DINOv2 ViT-B/14 | linear probe (C on familiar val) | 97.1 | 92.1 | **91.7** | 89.3 - 95.2 |
| ResNet-50 (ImageNet) | fine-tune last 2 blocks | 96.9 | 90.0 | **91.6** | 87.9 - 93.5 |
| CLIP ViT-B/16 | fine-tune last 2 blocks | 97.2 | 92.0 | **95.0** | 92.2 - 96.7 |
| SigLIP ViT-B/16 | fine-tune last 2 blocks | 97.6 | 93.6 | **95.1** | 93.0 - 96.6 |
| DINOv2 ViT-B/14 | fine-tune last 2 blocks | 98.0 | 95.7 | **95.7** | 91.5 - 97.7 |
<!-- /results:camelyon17 -->

### What the Camelyon17 results show

Numbers below are from `results/camelyon17_*.csv`. Differences are in test accuracy, with
95% intervals from resampling the same slides for both models (`camelyon17_paired.csv`).

1. **Every model loses accuracy at a new hospital.** With frozen features (linear
   probes) the drop from familiar to test hospital is 5 to 9 points; after partial
   fine-tuning it is 2 to 5. Pretraining on hundreds of millions of web images did not
   remove it.
2. **Frozen foundation-model features are not reliably better than an ImageNet
   ResNet-50.** The DINOv2 probe is highest (91.7 vs 88.8), but the paired interval for
   the gap, -0.9 to +11.4 points, includes zero. CLIP is +1.0 (-2.6 to +8.7) and SigLIP
   -1.4 (-6.0 to +0.5). The test hospital contributes only ten slides, and that is the
   honest width of the evidence.
3. **Foundation models pull ahead once they are allowed to adapt.** Fine-tuning only the
   last two blocks, for one epoch on 60,000 of the 302,000 training patches, gives
   DINOv2 95.7, SigLIP 95.1, CLIP 95.0 and ResNet-50 91.6. Against the fine-tuned
   ResNet-50 all three are clear of zero: DINOv2 +4.1 (+3.2 to +5.3), SigLIP +3.5
   (+1.3 to +6.1) and CLIP +3.4 (+1.6 to +5.0). The same fine-tuning gains ResNet-50
   itself only +2.9 over its probe (-0.7 to +8.8). What the large pretraining buys is
   features that adapt well, not features that are already robust.
4. **That gap survives retraining; the order within it does not.** Each fine-tuning was
   repeated with three seeds (a different training subset, batch order and starting
   head; `camelyon17_finetune_seed_summary.csv`). Mean test accuracy is DINOv2 94.9,
   SigLIP 94.7, CLIP 93.9 and ResNet-50 89.9, and the worst foundation-model run (93.3)
   is above the best ResNet-50 run (91.6). Which foundation model comes first changes
   from seed to seed, so the data supports "foundation models beat ResNet-50 after
   fine-tuning", not a ranking among them.
5. **Zero-shot classification does not work on pathology.** CLIP reaches 60 percent and
   SigLIP 50 percent, which is chance on a balanced task: SigLIP labels 99.9 percent of
   all patches as normal tissue. The web text these models learned from does not
   describe lymph node histology in a way the prompts can reach.
6. **WiSE-FT adds nothing here.** The best blends give almost all the weight to the
   trained probe (alpha 0.8 to 1.0) and change test accuracy by at most one point,
   because the zero-shot head it blends in is at chance.
7. **In-distribution accuracy does not predict new-hospital accuracy.** Across all twenty
   probe settings the correlation in probit space is 0.06. "Accuracy on the line" has
   nothing to work with when every model scores 96 to 97 percent on familiar data.
8. **The two new hospitals disagree about which model is best.** Among the probes, SigLIP
   is second-best on hospital 1 and worst on hospital 2; CLIP is the reverse. Choosing a
   model, or its regularisation, on one unseen hospital is weak evidence about another.
9. **Errors cluster on a few slides.** For every probe, WiSE-FT and fine-tuned model, 35
   to 64 percent of all test errors fall on just two of the ten test slides
   (`camelyon17_per_slide.csv`).
   85,054 test patches are much less evidence than the number suggests, which is why
   every interval here resamples slides.

Figures: `figures/camelyon17_main` (every backbone and method against the familiar
hospitals), `camelyon17_paired` (the paired differences behind points 2 and 3) and
`camelyon17_on_the_line` (point 7).

The table, the CSVs in `results/` and the figures in `figures/` are all regenerated from
the cached embeddings by one command, which runs each analysis script in turn:

```
python scripts/run_all.py camelyon17
```

Fine-tuned rows come from `python scripts/run_finetune.py camelyon17 <backbone>`, about
twenty minutes per backbone on a laptop GPU.

## Demo app

```
pip install streamlit
python -m streamlit run app/Home.py
```

Three pages. **Browse patches** shows real tissue patches from any hospital with two
chosen methods' answers, filtered to where they disagree, plus accuracy slide by slide.
**Results** shows the tables and figures, read from `results/` and `figures/`. **About**
explains the protocol in plain words. The app only reads saved predictions; it never runs
a model, so it starts in under a minute on a laptop.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m pytest
```

The first install line is for an NVIDIA GPU. Without one, drop it and pip installs the
CPU build; everything except embedding extraction and fine-tuning runs fine on CPU.

Then fetch and verify the data (about 15 minutes on a fast connection) and embed it:

```
python scripts/prepare_data.py camelyon17
python scripts/embed.py camelyon17 all
python scripts/prepare_data.py iwildcam
python scripts/embed.py iwildcam all
```

iWildCam has no mirror, so its roughly 200,000 images are fetched one file at a time from
the official bundle, in parallel. Rerunning after an interruption fetches only what is
missing.

Labels, hospitals and splits come from the official WILDS metadata. The images come from
a pinned copy on Hugging Face, because the official bundle only downloads as a single
10 GB stream that cannot resume. Every copied row is checked against the official
metadata, and a random sample of images is compared pixel by pixel with the official
files; preparation stops on any disagreement.

Images and embeddings live outside the repository, in `~/wilds-data` by default. Set the
`WILDS_DATA` environment variable to keep them somewhere else. Never put them in a synced
folder such as OneDrive.

## Conventions

- One short past-tense commit message per commit, no body. `scripts/check_commit_message.py`
  enforces this as a commit-msg hook.
- Every seed, split and hyperparameter lives in `configs/default.yaml`.
- Tests run locally with `python -m pytest`; there is no CI.
