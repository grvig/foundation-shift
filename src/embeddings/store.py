"""Embeddings on disk: one float16 matrix per (dataset, backbone), row i = metadata row i.

Every experiment after extraction reads these matrices instead of images, which is what
makes the project runnable on a laptop. That only works if row i really is image i, so the
store is guarded by a fingerprint: a hash of the ordered image paths and the backbone
settings. If the metadata, its order or the backbone changes, loading fails loudly rather
than pairing embeddings with the wrong labels - a mistake that would produce plausible,
wrong numbers with no visible symptom.

Layout under ``<data_root>/embeddings/<dataset>/<backbone>/``::

    manifest.json        fingerprint, row count, dimension, whether extraction finished
    shard_00000.npy ...  written during extraction, so a crash loses at most one shard
    embeddings.npy       the shards concatenated, written last

float16 halves the disk and memory cost. The models are run in float16 on the GPU anyway,
so storing float32 would only preserve rounding noise.
"""

import hashlib
import json

import numpy as np

MANIFEST = "manifest.json"
FINAL = "embeddings.npy"


def fingerprint(paths, settings):
    digest = hashlib.sha256()
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(b"\n")
        digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()[:16]


def store_dir(data_root, dataset, backbone):
    return data_root / "embeddings" / dataset / backbone


def shard_path(directory, index):
    return directory / ("shard_" + format(index, "05d") + ".npy")


def read_manifest(directory):
    path = directory / MANIFEST
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (MANIFEST + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    temporary.replace(directory / MANIFEST)


def save_array(path, array):
    """Write then rename, so a crash mid-write never leaves a truncated shard behind."""
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def check_fingerprint(directory, expected):
    manifest = read_manifest(directory)
    if manifest is None:
        return
    if manifest["fingerprint"] != expected:
        raise ValueError(
            "embeddings in " + str(directory) + " were made from different images or "
            "backbone settings (fingerprint " + manifest["fingerprint"] + ", expected "
            + expected + "). Delete that folder and extract again."
        )


def load_embeddings(data_root, dataset, backbone, paths, settings):
    """The finished matrix for these exact image paths, as float32."""
    directory = store_dir(data_root, dataset, backbone)
    manifest = read_manifest(directory)
    if manifest is None or not manifest.get("complete", False):
        raise FileNotFoundError(
            "no finished embeddings for " + dataset + "/" + backbone + "; run "
            "python scripts/embed.py " + dataset + " " + backbone
        )
    check_fingerprint(directory, fingerprint(paths, settings))
    array = np.load(directory / FINAL)
    if array.shape[0] != len(paths):
        raise ValueError("embedding rows do not match the metadata rows")
    return array.astype(np.float32)
