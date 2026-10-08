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
<!-- /results:camelyon17 -->

The table, the CSVs in `results/` and the figures in `figures/` are all regenerated from
the cached embeddings by one command, which runs each analysis script in turn:

```
python scripts/run_all.py camelyon17
```

Fine-tuned rows come from `python scripts/run_finetune.py camelyon17 <backbone>`, about
twenty minutes per backbone on a laptop GPU.

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
