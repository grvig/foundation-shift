"""Prepare a dataset: official metadata, mirror images, verification.

    python scripts/prepare_data.py camelyon17
    python scripts/prepare_data.py iwildcam

A dataset with a ``mirror_repo`` in the config (Camelyon17) goes through the steps below.
One without (iWildCam) downloads its metadata and then every image straight from the
official bundle, in parallel and resumably (see src/data/fetch.py).

1. Downloads the official ``metadata.csv`` from the WILDS bundle.
2. Downloads the mirror's parquet files at the pinned revision (resumable, and files
   already present are skipped).
3. Writes every image to its official path, checking each mirror row against the
   official metadata (see src/data/mirror.py).
4. Fetches a random sample of images from the official bundle and requires them to be
   pixel-identical to the files just written.
5. Deletes the parquet files unless ``--keep-mirror`` is given.

Steps 3 and 4 are why using a mirror is safe: every label, hospital and split comes from
the official file, and the images are shown to be the official images.
"""

import argparse
import io
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from PIL import Image

from src.config import load_config
from src.data.camelyon17 import METADATA_FILE
from src.data.datasets import load_metadata
from src.data.fetch import fetch_files
from src.data.iwildcam import CATEGORIES_FILE
from src.data.mirror import write_images


def fetch_official_file(bundle_url, relative_path):
    with urllib.request.urlopen(bundle_url + relative_path, timeout=120) as response:
        return response.read()


def fetch_mirror(settings, mirror_dir):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download

    names = []
    for name in HfApi().list_repo_files(settings["mirror_repo"], repo_type="dataset",
                                        revision=settings["mirror_revision"]):
        if name.endswith(".parquet"):
            names.append(name)
    paths = []
    for index, name in enumerate(sorted(names)):
        path = hf_hub_download(settings["mirror_repo"], name, repo_type="dataset",
                               revision=settings["mirror_revision"], local_dir=mirror_dir)
        print("  mirror file " + str(index + 1) + "/" + str(len(names)) + " ready")
        paths.append(Path(path))
    return paths


def spot_check(settings, metadata, dataset_dir, count, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(metadata), size=min(count, len(metadata)), replace=False)
    for number, row_id in enumerate(sorted(chosen)):
        relative = metadata["path"][int(row_id)]
        official = np.asarray(Image.open(io.BytesIO(
            fetch_official_file(settings["bundle_url"], relative))))
        local = np.asarray(Image.open(dataset_dir / relative))
        if official.shape != local.shape or not np.array_equal(official, local):
            raise ValueError("image " + relative + " differs from the official file")
        if (number + 1) % 25 == 0:
            print("  " + str(number + 1) + " images identical to the official files")


def prepare_from_bundle(settings, metadata, dataset_dir):
    """Datasets without a mirror: every image straight from the official bundle."""
    categories = dataset_dir / CATEGORIES_FILE
    if not categories.exists():
        try:
            categories.write_bytes(fetch_official_file(settings["bundle_url"],
                                                       CATEGORIES_FILE))
        except urllib.error.HTTPError:
            print("the bundle has no " + CATEGORIES_FILE + "; species will be numbered")
    print("fetching every image from the official bundle")
    fetch_files(settings["bundle_url"], list(metadata["path"]), dataset_dir,
                int(settings["fetch_workers"]))
    print("dataset ready at " + str(dataset_dir))


def main():
    parser = argparse.ArgumentParser(description="Download and verify a WILDS dataset.")
    parser.add_argument("dataset")
    parser.add_argument("--keep-mirror", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config.dataset(args.dataset)
    dataset_dir = config.data_root() / settings["extract_subdir"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / METADATA_FILE
    if not metadata_path.exists():
        print("downloading the official metadata")
        metadata_path.write_bytes(fetch_official_file(settings["bundle_url"], METADATA_FILE))
    metadata = load_metadata(config, args.dataset)
    print(str(len(metadata)) + " official rows")

    if "mirror_repo" not in settings:
        prepare_from_bundle(settings, metadata, dataset_dir)
        return

    mirror_dir = config.data_root() / "mirror" / args.dataset
    print("fetching the mirror at revision " + settings["mirror_revision"][:12])
    parquet_paths = fetch_mirror(settings, mirror_dir)

    print("writing images and checking every row against the official metadata")
    written = write_images(parquet_paths, metadata, dataset_dir)
    print(str(written) + " images written, every official row accounted for")

    print("comparing " + str(settings["spot_checks"]) + " random images with the official bundle")
    spot_check(settings, metadata, dataset_dir, int(settings["spot_checks"]), config.seed)
    print("all sampled images identical")

    if not args.keep_mirror:
        shutil.rmtree(mirror_dir)
        print("deleted the mirror files to free space")
    print("dataset ready at " + str(dataset_dir))


if __name__ == "__main__":
    main()
