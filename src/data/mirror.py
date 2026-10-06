"""Unpack a verified Hugging Face copy of a WILDS dataset into the official file layout.

The mirror stores images inside parquet files, one row per image, alongside the fields
that identify it. Each row is matched to its official metadata row by ``image_id`` and
every identifying field must agree; the image is then written to the exact path the
official archive uses. Everything downstream therefore sees the same files it would have
seen from the official download, and the metadata it reads is the official file, never
the mirror's.

Preparation fails unless every official row is written exactly once. Files are written
under a temporary name and renamed, so an interrupted run can be restarted and skips only
images that were finished.
"""

import numpy as np
import pyarrow.parquet as pq

from src.data.camelyon17 import mirror_mismatches

MIRROR_COLUMNS = ["image", "image_id", "patient", "node", "x_coord", "y_coord", "slide",
                  "center", "label"]


def write_images(parquet_paths, metadata, dataset_dir, log=print):
    """Write every mirror image to its official path; return how many were new."""
    seen = np.zeros(len(metadata), dtype=bool)
    written = 0
    made = set()
    for parquet_path in parquet_paths:
        problems = []
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=4096,
                                                                columns=MIRROR_COLUMNS):
            rows = batch.to_pylist()
            for row in rows:
                image_id = int(row["image_id"])
                if image_id < 0 or image_id >= len(metadata):
                    problems.append("image_id " + str(image_id) + " is not an official row")
                    continue
                if seen[image_id]:
                    problems.append("image_id " + str(image_id) + " appears twice")
                    continue
                seen[image_id] = True
                official = metadata.iloc[image_id]
                wrong = mirror_mismatches(official, row)
                if len(wrong) > 0:
                    problems.append("image_id " + str(image_id) + " disagrees on "
                                    + ", ".join(wrong))
                    continue
                target = dataset_dir / official["path"]
                if target.exists():
                    continue
                if target.parent not in made:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    made.add(target.parent)
                temporary = target.with_name(target.name + ".tmp")
                temporary.write_bytes(row["image"]["bytes"])
                temporary.replace(target)
                written = written + 1
        if len(problems) > 0:
            raise ValueError(str(parquet_path) + " does not match the official metadata ("
                             + str(len(problems)) + " problems), first: " + problems[0])
        log("  " + parquet_path.name + ": " + str(int(seen.sum())) + " of "
            + str(len(metadata)) + " rows checked")
    missing = int((~seen).sum())
    if missing > 0:
        raise ValueError(str(missing) + " official rows have no image in the mirror")
    return written
