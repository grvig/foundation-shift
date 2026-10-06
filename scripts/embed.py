"""Embed every image of a dataset with one backbone, or all of them.

    python scripts/embed.py camelyon17 clip_b16
    python scripts/embed.py camelyon17 all
    python scripts/embed.py camelyon17 resnet50 --limit 2000     # quick check

Finished embeddings are skipped, and an interrupted run resumes from its last finished
shard. ``--limit`` embeds only the first N images into a separate store, which is how to
check a backbone end to end in a minute before committing to the full run.
"""

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from src.config import load_config
from src.data.datasets import dataset_dir
from src.data.datasets import load_metadata
from src.embeddings.extract import extract
from src.embeddings.store import fingerprint
from src.embeddings.store import read_manifest
from src.embeddings.store import store_dir
from src.models.backbones import backbone_names
from src.models.backbones import backbone_settings
from src.models.backbones import load_encoder


def embed_one(config, dataset, backbone, limit, device):
    metadata = load_metadata(config, dataset)
    paths = list(metadata["path"])
    store_name = dataset
    if limit > 0:
        paths = paths[:limit]
        store_name = dataset + "_first" + str(limit)
    settings = backbone_settings(config, backbone)
    directory = store_dir(config.data_root(), store_name, backbone)
    manifest = read_manifest(directory)
    if manifest is not None and manifest.get("complete", False):
        print(store_name + "/" + backbone + " already finished; skipping")
        return

    embedding = config.section("embedding")
    encoder = load_encoder(config, backbone)
    started = time.time()
    extract(encoder, dataset_dir(config, dataset), paths, directory,
            fingerprint(paths, settings), int(embedding["shard_size"]),
            int(embedding["batch_size"]), int(embedding["num_workers"]), device)
    seconds = time.time() - started
    print(backbone + ": " + str(len(paths)) + " images in " + format(seconds / 60, ".1f")
          + " min (" + format(len(paths) / seconds, ".0f") + " images/s)")


def main():
    parser = argparse.ArgumentParser(description="Embed a dataset with pretrained backbones.")
    parser.add_argument("dataset")
    parser.add_argument("backbone", help="a backbone name from the config, or 'all'")
    parser.add_argument("--limit", type=int, default=0, help="embed only the first N images")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("no GPU found; embedding on the CPU will be very slow")

    names = [args.backbone]
    if args.backbone == "all":
        names = backbone_names(config)
    for name in names:
        embed_one(config, args.dataset, name, args.limit, device)


# Windows starts DataLoader workers by re-importing this file, so the work must only run
# under the main guard.
if __name__ == "__main__":
    main()
