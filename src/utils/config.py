"""Config loading with relative paths (no hardcoded local paths)."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
REQUIRED_KEYS = ("project", "paths", "datasets")


def load_config(path=None):
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    return cfg


def resolve_path(cfg, key):
    """Return absolute Path for a key under cfg['paths']."""
    try:
        rel = cfg["paths"][key]
    except KeyError as e:
        raise KeyError(f"Unknown path key: {key}") from e
    return PROJECT_ROOT / rel


def dataset_raw_dir(cfg, name):
    if name not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset: {name}")
    return resolve_path(cfg, "raw_dir") / cfg["datasets"][name]["raw_subdir"]
