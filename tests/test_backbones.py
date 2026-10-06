"""The registry, and the transform every backbone shares. Loading real weights is not
tested here: it downloads hundreds of megabytes per model."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.models.backbones import backbone_names
from src.models.backbones import backbone_settings
from src.models.backbones import normalisation_from
from src.models.backbones import supports_text
from src.models.backbones import uniform_transform


def test_the_four_backbones_are_registered():
    names = backbone_names(load_config())
    assert names == ["resnet50", "clip_b16", "siglip_b16", "dinov2_b14"]


def test_only_image_text_models_support_zero_shot():
    config = load_config()
    supported = []
    for name in backbone_names(config):
        if supports_text(config, name):
            supported.append(name)
    assert supported == ["clip_b16", "siglip_b16"]


def test_every_backbone_runs_at_the_same_resolution():
    config = load_config()
    for name in backbone_names(config):
        assert backbone_settings(config, name)["image_size"] == 224


def test_an_unknown_backbone_fails_loudly():
    with pytest.raises(ValueError):
        backbone_settings(load_config(), "vgg16")


def test_the_whole_patch_survives_the_transform():
    """A patch with a red left edge must still have it after resizing: no crop."""
    image = Image.new("RGB", (96, 96), (0, 0, 255))
    for x in range(8):
        for y in range(96):
            image.putpixel((x, y), (255, 0, 0))
    tensor = uniform_transform(224, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))(image)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert float(tensor[0, 112, 0]) > 0.9
    assert float(tensor[2, 112, 0]) < 0.1


def test_normalisation_is_read_from_the_library_transform():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])
    assert normalisation_from(preprocess) == ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    with pytest.raises(ValueError):
        normalisation_from(transforms.Compose([transforms.ToTensor()]))
