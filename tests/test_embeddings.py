"""Extraction and the embedding store, with a tiny stand-in network on generated images."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image
from torchvision import transforms

from src.embeddings.extract import extract
from src.embeddings.store import fingerprint
from src.embeddings.store import load_embeddings
from src.embeddings.store import store_dir
from src.models.backbones import Encoder

SETTINGS = {"library": "test", "name": "average-colour"}


class CountingEncoder(Encoder):
    """Embeds an image as its average colour and counts how many images it was given."""

    def __init__(self):
        network = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
        super().__init__("test", network, transform, self.count_and_encode, 3)
        self.seen = 0

    def count_and_encode(self, batch):
        self.seen = self.seen + batch.shape[0]
        return self.network(batch)


@pytest.fixture
def images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for index in range(7):
        name = "patch_" + str(index) + ".png"
        Image.new("RGB", (16, 16), (index * 30, 0, 255 - index * 30)).save(image_dir / name)
        paths.append(name)
    return image_dir, paths


def run(encoder, image_dir, paths, root):
    directory = store_dir(root, "toy", "test")
    extract(encoder, image_dir, paths, directory, fingerprint(paths, SETTINGS),
            shard_size=3, batch_size=2, num_workers=0, device=torch.device("cpu"),
            log=quiet)
    return directory


def quiet(message):
    pass


def test_rows_line_up_with_the_images(images, tmp_path):
    image_dir, paths = images
    run(CountingEncoder(), image_dir, paths, tmp_path)
    array = load_embeddings(tmp_path, "toy", "test", paths, SETTINGS)
    assert array.shape == (7, 3)
    assert array.dtype == np.float32
    reds = array[:, 0]
    assert np.all(np.diff(reds) > 0)
    assert array[3, 0] == pytest.approx(90 / 255, abs=1e-3)


def test_an_interrupted_run_resumes_from_its_shards(images, tmp_path):
    image_dir, paths = images
    encoder = CountingEncoder()
    directory = store_dir(tmp_path, "toy", "test")
    # Pretend a previous run finished the first shard and then crashed.
    first = CountingEncoder()
    run(first, image_dir, paths[:3], tmp_path / "scratch")
    shard = np.load(store_dir(tmp_path / "scratch", "toy", "test") / "embeddings.npy")
    directory.mkdir(parents=True)
    np.save(directory / "shard_00000.npy", shard.astype(np.float16))
    run(encoder, image_dir, paths, tmp_path)
    assert encoder.seen == 4


def test_embeddings_for_other_images_are_refused(images, tmp_path):
    image_dir, paths = images
    run(CountingEncoder(), image_dir, paths, tmp_path)
    reordered = list(reversed(paths))
    with pytest.raises(ValueError, match="different images"):
        load_embeddings(tmp_path, "toy", "test", reordered, SETTINGS)
    with pytest.raises(ValueError, match="different images"):
        load_embeddings(tmp_path, "toy", "test", paths, {"library": "other"})


def test_unfinished_embeddings_cannot_be_loaded(tmp_path):
    with pytest.raises(FileNotFoundError, match="embed.py"):
        load_embeddings(tmp_path, "toy", "test", ["a.png"], SETTINGS)


def test_the_fingerprint_depends_on_order_and_settings():
    base = fingerprint(["a", "b"], SETTINGS)
    assert base == fingerprint(["a", "b"], dict(SETTINGS))
    assert base != fingerprint(["b", "a"], SETTINGS)
    assert base != fingerprint(["a", "b"], {"library": "x"})
