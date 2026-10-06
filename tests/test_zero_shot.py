import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import load_config
from src.methods.zero_shot import class_text_embeddings
from src.methods.zero_shot import zero_shot_predict

# A stand-in text model: a sentence about "tumor" points along the second axis, anything
# else along the first, and each template adds a little of the third so they differ.
WORDS = ["normal", "tumor"]


def fake_tokenize(sentences):
    rows = []
    for sentence in sentences:
        rows.append([int("tumor" in sentence), len(sentence)])
    return torch.tensor(rows, dtype=torch.float32)


def fake_encode_text(tokens):
    vectors = torch.zeros((tokens.shape[0], 3))
    vectors[:, 0] = 1.0 - tokens[:, 0]
    vectors[:, 1] = tokens[:, 0]
    vectors[:, 2] = tokens[:, 1] / 1000.0
    return vectors


def test_class_vectors_are_unit_length_and_point_the_right_way():
    vectors = class_text_embeddings(fake_encode_text, fake_tokenize, ["normal tissue",
                                    "tumor tissue"], ["a {}.", "an image of {}."],
                                    torch.device("cpu"))
    assert vectors.shape == (2, 3)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert vectors[0, 0] > 0.9
    assert vectors[1, 1] > 0.9


def test_images_go_to_the_nearest_class_regardless_of_scale():
    text = np.array([[1.0, 0.0], [0.0, 1.0]])
    images = np.array([[50.0, 1.0], [0.1, 0.3], [-2.0, -1.0]])
    assert list(zero_shot_predict(images, text)) == [0, 1, 1]


def test_the_config_has_one_description_per_label():
    prompts = load_config().section("zero_shot")["camelyon17"]
    assert len(prompts["classes"]) == 2
    assert "tumor" in prompts["classes"][1]
    for template in prompts["templates"]:
        assert "{}" in template
