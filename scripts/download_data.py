"""Fetch, verify and extract a WILDS dataset into the data root.

    python scripts/download_data.py camelyon17

The archive is about 10 GB and cannot be resumed: CodaLab streams it chunked, with no
length header, and answers a range request with HTTP 400. Bytes go to ``<archive>.part``,
which is only renamed once complete, so an interrupted run never leaves something that
looks finished; a rerun starts again from zero. CodaLab publishes no checksum either, so
the finished archive must match the exact byte count the WILDS package records. Anything
else is a hard failure, because a truncated archive can extract without complaint and
silently drop images from some hospitals.

Extraction refuses paths that would land outside the target folder. The archive is then
deleted unless ``--keep-archive`` is given, which halves the disk space the dataset needs.
"""

import argparse
import ssl
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.data.datasets import locate_dataset_dir

CHUNK_SIZE = 1024 * 1024


def open_url(url):
    """Open with full certificate verification, retrying against certifi's roots."""
    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        import certifi

        print("system store could not verify the chain; retrying with certifi's roots")
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=context)


def download(url, archive_path, expected_bytes):
    partial = archive_path.with_name(archive_path.name + ".part")
    print("downloading " + format(expected_bytes / 1e9, ".1f") + " GB to " + str(partial))
    with open_url(url) as response:
        written = 0
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written = written + len(chunk)
                percent = 100.0 * written / expected_bytes
                sys.stdout.write("\r  " + format(written / 1e9, ".2f") + " GB  ("
                                 + format(percent, ".1f") + "%)")
                sys.stdout.flush()
    print("")
    check_size(partial, expected_bytes)
    partial.replace(archive_path)


def check_size(path, expected_bytes):
    observed = path.stat().st_size
    if observed != expected_bytes:
        raise ValueError(
            str(path) + " is " + str(observed) + " bytes, expected " + str(expected_bytes)
            + ". Delete it and rerun; if the size is wrong again, the published archive "
            "has changed and archive_bytes in configs/default.yaml needs looking at."
        )
    print("size ok: " + str(observed) + " bytes")


def extract(archive_path, target):
    print("extracting to " + str(target) + " (this takes a while)")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        # The "data" filter rejects absolute paths, ".." components and links that point
        # outside the target, so a malformed archive cannot write elsewhere on disk.
        archive.extractall(target, filter="data")


def main():
    parser = argparse.ArgumentParser(description="Download and extract a WILDS dataset.")
    parser.add_argument("dataset", help="dataset name, as in configs/default.yaml")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.dataset(args.dataset)
    data_root = config.data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / settings["archive_name"]
    target = data_root / settings["extract_subdir"]

    if archive_path.exists():
        print("archive already present at " + str(archive_path))
        check_size(archive_path, int(settings["archive_bytes"]))
    else:
        download(settings["url"], archive_path, int(settings["archive_bytes"]))

    extract(archive_path, target)
    dataset_dir = locate_dataset_dir(target)
    print("dataset ready at " + str(dataset_dir))
    if not args.keep_archive:
        archive_path.unlink()
        print("deleted the archive to free space")


if __name__ == "__main__":
    main()
