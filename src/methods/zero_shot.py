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


def load_text_side(config, backbone, dataset, device):
    """The class vectors for one backbone and dataset, computed from the config prompts."""
    import open_clip

    from src.models.backbones import backbone_settings

    settings = backbone_settings(config, backbone)
    if settings["library"] != "open_clip":
        raise ValueError(backbone + " has no text encoder, so it cannot classify zero-shot")
    prompts = config.section("zero_shot")[dataset]
    model = open_clip.create_model(settings["name"], pretrained=settings["pretrained"])
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(settings["name"])
    return class_text_embeddings(model.encode_text, tokenizer, prompts["classes"],
                                 prompts["templates"], device)
