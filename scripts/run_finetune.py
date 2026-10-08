"""Partially fine-tune one backbone and score it on every evaluation split.

    python scripts/run_finetune.py camelyon17 clip_b16
    python scripts/run_finetune.py camelyon17 clip_b16 --quick     # minutes, not an hour

Trains on a seeded subset of the training split (see fine_tuning in the config), then
predicts every id_val, ood_val and test image. Appends one row to
results/<dataset>_finetune.csv, in the same columns as the main table, and saves the
predictions where run_compare.py can find them.

``--quick`` trains on 512 images and evaluates 512 per split, writing to a separate
``_quick`` file, to check a backbone end to end before the real run.
"""

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.datasets import dataset_dir
from src.data.datasets import load_metadata
from src.embeddings.extract import ImageFiles
from src.evaluation.bootstrap import cluster_bootstrap
from src.evaluation.metrics import metric_function
from src.methods.finetune import Classifier
from src.methods.finetune import freeze_all_but
from src.methods.finetune import image_model
from src.methods.finetune import predict
from src.methods.finetune import train
from src.methods.finetune import trainable_names
from src.methods.linear_probe import EVALUATION_SPLITS
from src.models.backbones import backbone_settings
from src.models.backbones import load_encoder


class LabelledImages(ImageFiles):
    def __init__(self, image_dir, paths, labels, transform):
        super().__init__(image_dir, paths, transform)
        self.labels = list(labels)

    def __getitem__(self, index):
        return super().__getitem__(index), int(self.labels[index])


def sample_rows(rows, count, rng):
    if count >= len(rows):
        return np.sort(rows)
    return np.sort(rng.choice(rows, size=count, replace=False))


def main():
    parser = argparse.ArgumentParser(description="Partially fine-tune one backbone.")
    parser.add_argument("dataset")
    parser.add_argument("backbone")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    settings = dict(config.section("fine_tuning"))
    dataset_settings = config.dataset(args.dataset)
    metadata = load_metadata(config, args.dataset)
    labels = metadata["label"].to_numpy()
    splits = metadata["split"].to_numpy()
    paths = metadata["path"].to_numpy()
    image_dir = dataset_dir(config, args.dataset)
    rng = config.fresh_rng()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_count = int(settings["train_images"])
    evaluate_limit = 0
    suffix = ""
    if args.quick:
        train_count = 512
        evaluate_limit = 512
        suffix = "_quick"

    encoder = load_encoder(config, args.backbone)
    library = backbone_settings(config, args.backbone)["library"]
    backbone = image_model(encoder, library)
    freeze_all_but(backbone, trainable_names(backbone, settings["trainable_blocks"]))
    model = Classifier(backbone, encoder.dimension, int(labels.max()) + 1)

    chosen = sample_rows(np.flatnonzero(splits == "train"), train_count, rng)
    workers = int(config.section("embedding")["num_workers"])
    loader = DataLoader(LabelledImages(image_dir, paths[chosen], labels[chosen],
                                       encoder.transform),
                        batch_size=int(settings["batch_size"]), shuffle=True,
                        num_workers=workers, generator=torch.Generator().manual_seed(config.seed))
    started = time.time()
    print(args.backbone + ": training on " + str(len(chosen)) + " images")
    train(model, loader, settings, device)

    metric = metric_function(dataset_settings["metric"])
    method = "finetune_last" + str(settings["trainable_blocks"])
    row = {"dataset": args.dataset, "backbone": args.backbone, "method": method,
           "selection": "none", "c": float("nan")}
    predictions = np.full(len(metadata), -1, dtype=np.int16)
    for split in EVALUATION_SPLITS:
        rows = np.flatnonzero(splits == split)
        if evaluate_limit > 0:
            rows = sample_rows(rows, evaluate_limit, rng)
        print(args.backbone + ": predicting " + str(len(rows)) + " " + split + " images")
        evaluation = DataLoader(ImageFiles(image_dir, paths[rows], encoder.transform),
                                batch_size=128, shuffle=False, num_workers=workers)
        predictions[rows] = predict(model, evaluation, device)
        row[split] = metric(labels[rows], predictions[rows])
    test = predictions >= 0
    test = test & (splits == "test")
    bootstrap = config.section("bootstrap")
    clusters = metadata[dataset_settings["cluster_column"]].to_numpy()
    _, row["test_low"], row["test_high"] = cluster_bootstrap(
        labels[test], predictions[test], clusters[test], dataset_settings["metric"],
        int(bootstrap["resamples"]), config.fresh_rng())
    row["minutes"] = (time.time() - started) / 60.0

    if not args.quick:
        directory = config.data_root() / "predictions" / args.dataset
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / (args.backbone + "_" + method + "_none.npy"), predictions)
    path = config.path("results_dir") / (args.dataset + "_finetune" + suffix + ".csv")
    table = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[previous["backbone"] != args.backbone]
        table = pd.concat([previous, table], ignore_index=True)
    table.to_csv(path, index=False, float_format="%.4f")
    print(pd.DataFrame([row]).to_string(index=False))


# Windows starts DataLoader workers by re-importing this file.
if __name__ == "__main__":
    main()
