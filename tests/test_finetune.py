"""Which parameters train, and that training moves only those, on tiny stand-in models."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.methods.finetune import Classifier
from src.methods.finetune import freeze_all_but
from src.methods.finetune import predict
from src.methods.finetune import train
from src.methods.finetune import trainable_names


class TinyViT(torch.nn.Module):
    """Named like a timm ViT: a stem, numbered blocks, then a final norm."""

    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(12, 4)
        self.blocks = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4),
                                          torch.nn.Linear(4, 4))
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, images):
        return self.norm(self.blocks(self.patch_embed(images.flatten(1))))


class TinyResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(2, 2)
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.layer4 = torch.nn.Sequential(torch.nn.Module(), torch.nn.Module())
        self.layer4[0].conv1 = torch.nn.Linear(2, 2)
        self.layer4[1].conv1 = torch.nn.Linear(2, 2)


def test_the_last_blocks_and_everything_after_them_train():
    names = trainable_names(TinyViT(), 1)
    assert names == ["blocks.2.weight", "blocks.2.bias", "norm.weight", "norm.bias"]


def test_a_conv1_inside_a_block_is_judged_by_its_block():
    names = trainable_names(TinyResNet(), 1)
    assert names == ["layer4.1.conv1.weight", "layer4.1.conv1.bias"]


def test_a_model_without_blocks_is_refused():
    with pytest.raises(ValueError):
        trainable_names(torch.nn.Linear(2, 2), 1)


def test_training_moves_only_the_trainable_parameters():
    torch.manual_seed(0)
    backbone = TinyViT()
    freeze_all_but(backbone, trainable_names(backbone, 1))
    model = Classifier(backbone, 4, 2)
    stem_before = backbone.patch_embed.weight.detach().clone()
    tail_before = backbone.blocks[2].weight.detach().clone()

    images = torch.randn(64, 3, 2, 2)
    # The label is readable from what the frozen layers pass on, so the trainable tail
    # can learn it; a label the frozen stem had discarded would be unlearnable.
    with torch.no_grad():
        labels = (backbone.blocks[1](backbone.blocks[0](
            backbone.patch_embed(images.flatten(1))))[:, 0] > 0).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels),
                                         batch_size=16)
    settings = {"epochs": 30, "backbone_lr": 1e-2, "head_lr": 1e-1, "weight_decay": 0.0}
    losses = train(model, loader, settings, torch.device("cpu"), log=quiet)

    assert torch.equal(backbone.patch_embed.weight, stem_before)
    assert not torch.equal(backbone.blocks[2].weight, tail_before)
    assert np.mean(losses[-4:]) < np.mean(losses[:4])
    predictions = predict(model, torch.utils.data.DataLoader(images, batch_size=16),
                          torch.device("cpu"))
    assert predictions.shape == (64,)
    assert np.mean(predictions == labels.numpy()) > 0.8


def quiet(message):
    pass
