from pathlib import Path

import pytest

from src.config import DATA_ROOT_VARIABLE
from src.config import load_config


def test_the_default_config_loads():
    config = load_config()
    assert config.seed == 20261005


def test_data_root_defaults_to_the_home_directory(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_VARIABLE, raising=False)
    root = load_config().data_root()
    assert root == Path.home() / "wilds-data"


def test_data_root_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_VARIABLE, raising=False)
    config = load_config()
    root = config.data_root().resolve()
    assert config.project_root.resolve() not in root.parents


def test_the_environment_variable_overrides_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_VARIABLE, str(tmp_path))
    assert load_config().data_root() == tmp_path


def test_a_missing_dataset_section_fails_loudly():
    with pytest.raises(KeyError):
        load_config().dataset("no-such-dataset")
