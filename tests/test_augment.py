import pickle
import random

import pytest

pytest.importorskip("torchvision")
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.methods.augment import RightAngleTurn
from src.methods.augment import training_transform


def marked_patch():
    image = Image.new("RGB", (8, 8), (200, 150, 200))
    image.putpixel((0, 0), (0, 0, 0))
    return image


def test_turns_are_exact_right_angles():
    random.seed(0)
    seen = set()
    for _ in range(40):
        turned = RightAngleTurn()(marked_patch())
        assert turned.size == (8, 8)
        for corner in [(0, 0), (7, 0), (0, 7), (7, 7)]:
            if turned.getpixel(corner) == (0, 0, 0):
                seen.add(corner)
    assert seen == {(0, 0), (7, 0), (0, 7), (7, 7)}


def test_the_evaluation_transform_runs_last_and_unchanged():
    evaluation = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    settings = load_config().section("augmentation")
    augmented = training_transform(evaluation, settings)
    assert augmented.transforms[-1] is evaluation
    assert tuple(augmented(marked_patch()).shape) == (3, 16, 16)


def test_the_transform_survives_pickling_for_windows_workers():
    evaluation = transforms.Compose([transforms.ToTensor()])
    augmented = training_transform(evaluation, load_config().section("augmentation"))
    assert pickle.loads(pickle.dumps(augmented)) is not None


def test_flips_can_be_switched_off():
    settings = dict(load_config().section("augmentation"))
    settings["flips"] = False
    augmented = training_transform(transforms.ToTensor(), settings)
    assert len(augmented.transforms) == 2
