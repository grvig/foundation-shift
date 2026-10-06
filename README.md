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
```

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
