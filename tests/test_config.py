import pytest
from src.utils.config import load_config, resolve_path, dataset_raw_dir, PROJECT_ROOT
from src.utils.seed import set_global_seed


def test_load_config_has_sections():
    cfg = load_config()
    assert {"project", "paths", "datasets"} <= set(cfg)


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_invalid_config_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("project: {}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_paths_are_relative_to_project_root():
    cfg = load_config()
    assert resolve_path(cfg, "raw_dir") == PROJECT_ROOT / "data/raw"
    with pytest.raises(KeyError):
        resolve_path(cfg, "nope")


def test_dataset_namespaces_are_separate():
    cfg = load_config()
    ns = [v["namespace"] for v in cfg["datasets"].values()]
    dirs = {dataset_raw_dir(cfg, d) for d in cfg["datasets"]}
    assert len(set(ns)) == len(ns) == 3
    assert len(dirs) == 3
    with pytest.raises(KeyError):
        dataset_raw_dir(cfg, "unknown")


def test_seed_reproducible():
    import numpy as np
    set_global_seed(1); a = np.random.rand(3)
    set_global_seed(1); b = np.random.rand(3)
    assert (a == b).all()
