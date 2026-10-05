"""Configuration loading and global seeding.

Every script starts by calling :func:`load_config`. The seed lives in exactly one place
(``configs/default.yaml``) so a run can be reproduced from the config file alone, and no
other module is allowed to seed a generator on its own.

Data lives outside the repository. The images are tens of gigabytes, and the repository
sits in a synced folder on at least one team member's machine; a dataset inside it would be
uploaded to the cloud. ``data_root`` therefore defaults to a folder in the home directory,
and the ``WILDS_DATA`` environment variable overrides it for anyone who keeps data
elsewhere, such as a second drive.
"""

import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DATA_ROOT_VARIABLE = "WILDS_DATA"

OUTPUT_DIR_KEYS = ["results_dir", "figures_dir"]


class Config:
    """Read-only view over the parsed YAML."""

    def __init__(self, values, project_root):
        self.values = values
        self.project_root = Path(project_root)
        self.seed = int(values["seed"])

    def section(self, name):
        if name not in self.values:
            raise KeyError("missing config section: " + name)
        return self.values[name]

    def path(self, name):
        """A repository-relative output path such as results_dir."""
        paths = self.section("paths")
        if name not in paths:
            raise KeyError("missing path entry: " + name)
        return self.project_root / paths[name]

    def data_root(self):
        """Where images and embeddings live: the environment variable wins over the file."""
        override = os.environ.get(DATA_ROOT_VARIABLE, "").strip()
        if override != "":
            return Path(override).expanduser()
        return Path(self.section("paths")["data_root"]).expanduser()

    def dataset(self, name):
        datasets = self.section("datasets")
        if name not in datasets:
            raise KeyError("missing dataset section: " + name)
        return dict(datasets[name])

    def fresh_rng(self):
        """A generator at the seeded starting state, so repeated calls give identical draws."""
        return np.random.default_rng(self.seed)

    def ensure_output_dirs(self):
        for key in OUTPUT_DIR_KEYS:
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path=None):
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("config file not found: " + str(path))
    with open(path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    config = Config(values, PROJECT_ROOT)
    seed_everything(config.seed)
    return config


def seed_everything(seed):
    """Seed the two global generators the project touches."""
    random.seed(seed)
    np.random.seed(seed)
