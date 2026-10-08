"""The parallel fetcher, against an in-memory stand-in for the bundle server."""

import io
import urllib.error

import pytest
from PIL import Image

from src.data.fetch import fetch_files


def jpeg(colour):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeServer:
    """Serves files from a dict; can fail a path a set number of times first."""

    def __init__(self, files, flaky=None):
        self.files = files
        self.flaky = dict(flaky or {})
        self.requests = []

    def __call__(self, url):
        path = url.split("bundle/")[1]
        self.requests.append(path)
        if self.flaky.get(path, 0) > 0:
            self.flaky[path] = self.flaky[path] - 1
            raise urllib.error.URLError("connection reset")
        if path not in self.files:
            raise urllib.error.URLError("not found")
        return io.BytesIO(self.files[path])


FILES = {"train/a.jpg": jpeg((255, 0, 0)), "train/b.jpg": jpeg((0, 255, 0))}


def quiet(message):
    pass


def test_every_file_is_written_where_the_metadata_says(tmp_path):
    server = FakeServer(FILES)
    assert fetch_files("x/bundle/", list(FILES), tmp_path, 2, server, quiet) == 2
    assert Image.open(tmp_path / "train/a.jpg").getpixel((4, 4))[0] > 200
    assert len(list(tmp_path.rglob("*.tmp"))) == 0


def test_a_rerun_fetches_only_missing_files(tmp_path):
    fetch_files("x/bundle/", list(FILES), tmp_path, 2, FakeServer(FILES), quiet)
    (tmp_path / "train/b.jpg").unlink()
    server = FakeServer(FILES)
    assert fetch_files("x/bundle/", list(FILES), tmp_path, 2, server, quiet) == 1
    assert server.requests == ["train/b.jpg"]


def test_a_dropped_connection_is_retried(tmp_path):
    server = FakeServer(FILES, flaky={"train/a.jpg": 2})
    fetch_files("x/bundle/", list(FILES), tmp_path, 1, server, quiet)
    assert (tmp_path / "train/a.jpg").exists()


def test_a_file_that_never_arrives_fails_the_run(tmp_path):
    server = FakeServer(FILES)
    with pytest.raises(ValueError, match="1 files could not be fetched"):
        fetch_files("x/bundle/", list(FILES) + ["train/missing.jpg"], tmp_path, 2,
                    server, quiet)
    assert (tmp_path / "train/a.jpg").exists()


def test_a_corrupt_image_is_not_saved(tmp_path):
    server = FakeServer({"train/a.jpg": b"not an image"})
    with pytest.raises(ValueError):
        fetch_files("x/bundle/", ["train/a.jpg"], tmp_path, 1, server, quiet)
    assert not (tmp_path / "train/a.jpg").exists()
