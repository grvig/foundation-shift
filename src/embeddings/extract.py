"""Run a backbone over every image once and store the vectors.

This is the only step that needs a GPU for inference. Images are decoded by DataLoader
worker processes while the GPU embeds the previous batch, and the network runs under
float16 autocast, which roughly doubles throughput on a consumer card and fits a ViT-B in
4 GB with room to spare.
"""

import contextlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.embeddings.store import FINAL
from src.embeddings.store import check_fingerprint
from src.embeddings.store import save_array
from src.embeddings.store import shard_path
from src.embeddings.store import write_manifest


class ImageFiles(Dataset):
    """Images by relative path. Defined at module level so Windows workers can pickle it."""

    def __init__(self, image_dir, paths, transform):
        self.image_dir = image_dir
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.image_dir / self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def precision_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def embed_shard(encoder, image_dir, paths, batch_size, num_workers, device):
    loader = DataLoader(ImageFiles(image_dir, paths, encoder.transform),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    parts = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with precision_context(device):
                vectors = encoder.encode(batch)
            parts.append(vectors.float().cpu().numpy().astype(np.float16))
    return np.concatenate(parts, axis=0)


def extract(encoder, image_dir, paths, directory, expected_fingerprint, shard_size,
            batch_size, num_workers, device, log=print):
    """Embed ``paths`` into ``directory``, skipping shards a previous run finished."""
    check_fingerprint(directory, expected_fingerprint)
    manifest = {"fingerprint": expected_fingerprint, "rows": len(paths),
                "dimension": encoder.dimension, "shard_size": shard_size,
                "complete": False}
    write_manifest(directory, manifest)
    encoder.network.to(device)

    shard_count = (len(paths) + shard_size - 1) // shard_size
    shards = []
    for index in range(shard_count):
        start = index * shard_size
        stop = min(start + shard_size, len(paths))
        path = shard_path(directory, index)
        if path.exists() and np.load(path, mmap_mode="r").shape[0] == stop - start:
            log("shard " + str(index + 1) + "/" + str(shard_count) + " already done")
            shards.append(path)
            continue
        log("shard " + str(index + 1) + "/" + str(shard_count) + ": rows "
            + str(start) + "-" + str(stop))
        array = embed_shard(encoder, image_dir, paths[start:stop], batch_size,
                            num_workers, device)
        save_array(path, array)
        shards.append(path)

    parts = []
    for path in shards:
        parts.append(np.load(path))
    save_array(directory / FINAL, np.concatenate(parts, axis=0))
    manifest["complete"] = True
    write_manifest(directory, manifest)
    for path in shards:
        path.unlink()
    log("wrote " + str(directory / FINAL))
