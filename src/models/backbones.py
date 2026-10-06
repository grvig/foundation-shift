"""One place that knows how to load every pretrained image model.

Each backbone is loaded as an :class:`Encoder`: the network, the image transform it
expects, and a function from a batch of images to one vector per image.

**Every backbone sees the image the same way.** Each library ships its own preprocessing,
and they disagree: timm's ImageNet transform resizes to 256 and centre-crops 224, cutting
an eighth off every side, while open_clip resizes straight to 224. Camelyon17 patches are
only 96 pixels, so that crop would hide tissue from some models and not others, and part
of any accuracy difference would be the crop rather than the model. Here every image is
resized whole to the model's input size with bicubic interpolation, and only the
normalisation constants (mean and standard deviation) come from the model.

**What "the embedding" means per family.** For the image-text models (CLIP, SigLIP) it is
the output of ``encode_image``, the projected vector that lives in the same space as the
text embeddings, which is what zero-shot classification needs. For the timm models it is
the pooled feature with the classifier head removed: global-average-pooled for ResNet-50,
the class token for DINOv2.
"""

from torchvision import transforms


class Encoder:
    def __init__(self, name, network, transform, encode, dimension):
        self.name = name
        self.network = network
        self.transform = transform
        self.encode = encode
        self.dimension = dimension


def uniform_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


def normalisation_from(preprocess):
    """The mean and std of the Normalize step inside a library's own transform."""
    for step in preprocess.transforms:
        if isinstance(step, transforms.Normalize):
            return tuple(step.mean), tuple(step.std)
    raise ValueError("the library transform has no Normalize step")


def load_timm(name, settings):
    import timm

    image_size = int(settings["image_size"])
    options = {"pretrained": True, "num_classes": 0}
    if "dinov2" in settings["name"]:
        options["img_size"] = image_size
    network = timm.create_model(settings["name"], **options)
    data_config = timm.data.resolve_model_data_config(network)
    transform = uniform_transform(image_size, data_config["mean"], data_config["std"])
    return Encoder(name, network, transform, network, int(network.num_features))


def load_open_clip(name, settings):
    import open_clip

    network, _, preprocess = open_clip.create_model_and_transforms(
        settings["name"], pretrained=settings["pretrained"])
    mean, std = normalisation_from(preprocess)
    transform = uniform_transform(int(settings["image_size"]), mean, std)
    # The shared image-text space size. Read from the model config because the vision
    # tower's own attributes differ: SigLIP wraps a timm model with no output_dim.
    dimension = int(open_clip.get_model_config(settings["name"])["embed_dim"])
    return Encoder(name, network, transform, network.encode_image, dimension)


LOADERS = {"timm": load_timm, "open_clip": load_open_clip}


def backbone_names(config):
    return list(config.section("backbones").keys())


def backbone_settings(config, name):
    backbones = config.section("backbones")
    if name not in backbones:
        raise ValueError("unknown backbone: " + name + "; known: " + ", ".join(backbones))
    return dict(backbones[name])


def load_encoder(config, name):
    """Download (first time only) and build one backbone in evaluation mode."""
    settings = backbone_settings(config, name)
    library = settings["library"]
    if library not in LOADERS:
        raise ValueError("unknown library for " + name + ": " + library)
    encoder = LOADERS[library](name, settings)
    encoder.network.eval()
    return encoder


def supports_text(config, name):
    """Only the image-text models can classify zero-shot from a written description."""
    return backbone_settings(config, name)["library"] == "open_clip"
