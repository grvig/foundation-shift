"""Find an extracted dataset and load its metadata, by the name used in the config."""

from src.data import camelyon17
from src.data import iwildcam

MARKER_FILE = "metadata.csv"


def locate_dataset_dir(target):
    """The folder holding metadata.csv: the target itself, or one level below it.

    WILDS archives differ in whether they wrap their contents in a top-level folder, so
    both layouts are accepted.
    """
    if (target / MARKER_FILE).exists():
        return target
    if target.exists():
        for child in sorted(target.iterdir()):
            if child.is_dir() and (child / MARKER_FILE).exists():
                return child
    raise FileNotFoundError("no " + MARKER_FILE + " found under " + str(target)
                            + "; run scripts/prepare_data.py first")


def dataset_dir(config, name):
    settings = config.dataset(name)
    return locate_dataset_dir(config.data_root() / settings["extract_subdir"])


def load_metadata(config, name):
    settings = config.dataset(name)
    directory = dataset_dir(config, name)
    if name == "camelyon17":
        return camelyon17.load_metadata(directory, settings["val_center"],
                                        settings["test_center"])
    if name == "iwildcam":
        return iwildcam.load_metadata(directory)
    raise ValueError("no metadata loader for dataset: " + name)
