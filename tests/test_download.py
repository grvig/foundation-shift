"""The parts of the downloader that can be checked without a network."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_data import check_size
from download_data import extract
from download_data import locate_dataset_dir


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_a_wrong_size_is_a_hard_failure(tmp_path):
    path = tmp_path / "archive.tar.gz"
    path.write_bytes(b"x" * 10)
    check_size(path, 10)
    with pytest.raises(ValueError, match="expected 11"):
        check_size(path, 11)


def test_metadata_is_found_at_the_top_level(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive, {"metadata.csv": "a,b\n", "patches/p.png": "x"})
    target = tmp_path / "out"
    extract(archive, target)
    assert locate_dataset_dir(target) == target


def test_metadata_is_found_one_level_down(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive, {"camelyon17_v1.0/metadata.csv": "a,b\n"})
    target = tmp_path / "out"
    extract(archive, target)
    assert locate_dataset_dir(target) == target / "camelyon17_v1.0"


def test_a_path_escaping_the_target_is_refused(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive, {"../escaped.txt": "x"})
    with pytest.raises(tarfile.TarError):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_an_archive_without_metadata_fails_loudly(tmp_path):
    archive = tmp_path / "a.tar.gz"
    make_archive(archive, {"readme.txt": "x"})
    target = tmp_path / "out"
    extract(archive, target)
    with pytest.raises(FileNotFoundError):
        locate_dataset_dir(target)
