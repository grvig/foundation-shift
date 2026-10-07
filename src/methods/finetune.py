"""Partial fine-tuning: train the last few blocks of an image model and a linear head.

**Which parameters train.** Every backbone here is a stack of repeated blocks (``blocks.N``
in timm ViTs, ``resblocks.N`` in CLIP, ``layer4.N`` for the last ResNet stage) with a stem
before and a pooling or projection layer after. The last ``trainable_blocks`` blocks train,
as does everything after the blocks; the stem and earlier blocks are frozen. The rule is
decided by parameter name, checking the block pattern first so a ``conv1`` inside a
ResNet block is judged by its block rather than mistaken for the stem.

**The image model.** For open_clip backbones only ``network.visual`` is used: the full
model also holds a text transformer whose ``resblocks`` would otherwise match the rule.
``visual`` is exactly what ``encode_image`` runs, so fine-tuning starts from the same
features the linear probe and zero-shot results used.

Training uses float16 autocast with a gradient scaler, and two learning rates: a small one
for the pretrained blocks so they move gently, a larger one for the new head.
"""

import contextlib
import re

import numpy as np
import torch

BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:blocks|resblocks|layer4)\.(\d+)\.")
STEM_PATTERN = re.compile(r"(?:^|\.)(?:patch_embed|pos_embed|cls_token|reg_token|conv1|bn1|"
                          r"class_embedding|positional_embedding|ln_pre|norm_pre|layer1|"
                          r"layer2|layer3|stem)(?:\.|$)")


def image_model(encoder, library):
    if library == "open_clip":
        return encoder.network.visual
    return encoder.network


def trainable_names(module, trainable_blocks):
    """Names of the parameters that train under the rule in the module docstring."""
    names = []
    indices = []
    for name, _ in module.named_parameters():
        names.append(name)
        match = BLOCK_PATTERN.search(name)
        if match is not None:
            indices.append(int(match.group(1)))
    if len(indices) == 0:
        raise ValueError("found no repeated blocks to fine-tune")
    first = max(indices) - int(trainable_blocks) + 1
    chosen = []
    for name in names:
        match = BLOCK_PATTERN.search(name)
        if match is not None:
            if int(match.group(1)) >= first:
                chosen.append(name)
            continue
        if STEM_PATTERN.search(name) is not None:
            continue
        chosen.append(name)
    return chosen


def freeze_all_but(module, names):
    keep = set(names)
    for name, parameter in module.named_parameters():
        parameter.requires_grad = name in keep


class Classifier(torch.nn.Module):
    def __init__(self, backbone, dimension, classes):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(dimension, classes)

    def forward(self, images):
        return self.head(self.backbone(images).float())


def precision(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_optimiser(model, settings):
    backbone_parameters = []
    for parameter in model.backbone.parameters():
        if parameter.requires_grad:
            backbone_parameters.append(parameter)
    return torch.optim.AdamW([
        {"params": backbone_parameters, "lr": float(settings["backbone_lr"])},
        {"params": list(model.head.parameters()), "lr": float(settings["head_lr"])},
    ], weight_decay=float(settings["weight_decay"]))


def train(model, loader, settings, device, log=print):
    model.to(device)
    model.train()
    optimiser = make_optimiser(model, settings)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    for epoch in range(int(settings["epochs"])):
        for step, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with precision(device):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(float(loss.detach()))
            if (step + 1) % 100 == 0:
                log("  epoch " + str(epoch + 1) + " step " + str(step + 1) + " loss "
                    + format(float(np.mean(losses[-100:])), ".4f"))
    return losses


def predict(model, loader, device):
    model.to(device)
    model.eval()
    parts = []
    with torch.no_grad():
        for images in loader:
            with precision(device):
                logits = model(images.to(device, non_blocking=True))
            parts.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(parts)
