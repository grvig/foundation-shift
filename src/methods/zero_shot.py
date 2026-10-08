"""Zero-shot classification: no training, only written descriptions of each class.

An image-text model maps images and sentences into the same space. Each class gets one
vector: its description is slotted into several templates ("a histopathology image of
{}."), each sentence is embedded, and the unit-length results are averaged and normalised
again. An image is assigned the class whose vector is closest by cosine similarity.

Averaging over templates (prompt ensembling) makes the result depend less on the exact
wording of any one sentence. The prompts are fixed in the config before results are seen.
"""

import numpy as np
import torch

from src.methods.linear_probe import l2_normalise


def class_text_embeddings(encode_text, tokenize, classes, templates, device):
    """One unit-length vector per class, averaged over every template."""
    vectors = []
    with torch.no_grad():
        for description in classes:
            sentences = []
            for template in templates:
                sentences.append(template.format(description))
            tokens = tokenize(sentences).to(device)
            embedded = encode_text(tokens).float().cpu().numpy()
            mean = l2_normalise(embedded).mean(axis=0)
            vectors.append(mean / np.linalg.norm(mean))
    return np.stack(vectors)


def zero_shot_scores(image_features, text_embeddings):
    """Cosine similarity of every image to every class, shape (images, classes)."""
    return l2_normalise(image_features) @ text_embeddings.T


def zero_shot_predict(image_features, text_embeddings):
    return np.argmax(zero_shot_scores(image_features, text_embeddings), axis=1)


class TextHead:
    """A zero-shot classifier: class vectors plus the model's learned logit scale and bias.

    Predictions only need the class vectors (scale and bias do not change the argmax),
    but blending with another classifier's logits, as WiSE-FT does, needs the logits at
    the scale the model was trained to produce. SigLIP also learns a bias; CLIP has none.
    """

    def __init__(self, vectors, scale, bias):
        self.vectors = vectors
        self.scale = float(scale)
        self.bias = float(bias)

    def logits(self, image_features):
        return self.scale * zero_shot_scores(image_features, self.vectors) + self.bias


def class_descriptions(config, dataset):
    """The text slotted into each template, one entry per label in label order.

    Either listed in the config (Camelyon17's two classes) or, for a dataset with many
    classes, read from the dataset's own category names with any configured overrides.
    """
    prompts = config.section("zero_shot")[dataset]
    if "classes" in prompts:
        return list(prompts["classes"])
    if prompts.get("classes_from") != "categories":
        raise ValueError("zero_shot." + dataset + " needs classes or classes_from")
    from src.data import iwildcam
    from src.data.datasets import dataset_dir
    from src.data.datasets import load_metadata

    labels = load_metadata(config, dataset)["label"]
    overrides = dict(prompts.get("class_overrides") or {})
    descriptions = []
    for name in iwildcam.class_names(dataset_dir(config, dataset), labels):
        descriptions.append(overrides.get(name, name))
    return descriptions


def load_text_head(config, backbone, dataset, device):
    """The zero-shot classifier for one backbone and dataset, from the config prompts."""
    import open_clip

    from src.models.backbones import backbone_settings

    settings = backbone_settings(config, backbone)
    if settings["library"] != "open_clip":
        raise ValueError(backbone + " has no text encoder, so it cannot classify zero-shot")
    prompts = config.section("zero_shot")[dataset]
    model = open_clip.create_model(settings["name"], pretrained=settings["pretrained"])
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(settings["name"])
    vectors = class_text_embeddings(model.encode_text, tokenizer,
                                    class_descriptions(config, dataset),
                                    prompts["templates"], device)
    scale = float(model.logit_scale.exp().item())
    bias = 0.0
    if getattr(model, "logit_bias", None) is not None:
        bias = float(model.logit_bias.item())
    return TextHead(vectors, scale, bias)


def load_text_side(config, backbone, dataset, device):
    """Only the class vectors, which is all a zero-shot prediction needs."""
    return load_text_head(config, backbone, dataset, device).vectors
