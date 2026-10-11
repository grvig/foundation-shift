"""Training-time augmentation for tissue patches: colour jitter, flips and right-angle turns.

What differs most between Camelyon17's hospitals is the staining: the same tissue comes
out pinker or bluer, lighter or darker, depending on the lab's protocol and scanner.
Randomly perturbing brightness, contrast, saturation and hue during training is the
standard cheap defence, because it stops a model leaning on the exact colours of the
training hospitals. Flips and 90-degree rotations cost nothing and are label-preserving
for tissue, which has no up or down.

The augmentations run on the PIL image before the backbone's own resize and
normalisation, so the evaluation transform is unchanged and predictions on validation
and test images are made exactly as for every other method.
"""

import random

from torchvision import transforms


class RightAngleTurn:
    """Rotate by 0, 90, 180 or 270 degrees, chosen uniformly.

    A class rather than a lambda so DataLoader workers on Windows can pickle it.
    """

    def __call__(self, image):
        turns = random.randint(0, 3)
        if turns == 0:
            return image
        return image.rotate(90 * turns)


def training_transform(evaluation_transform, settings):
    steps = []
    if settings.get("flips", False):
        steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.RandomVerticalFlip())
        steps.append(RightAngleTurn())
    steps.append(transforms.ColorJitter(brightness=float(settings["brightness"]),
                                        contrast=float(settings["contrast"]),
                                        saturation=float(settings["saturation"]),
                                        hue=float(settings["hue"])))
    steps.append(evaluation_transform)
    return transforms.Compose(steps)
