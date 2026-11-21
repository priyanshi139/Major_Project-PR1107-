"""Run: python scripts/check_environment.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
from src.utils.config import load_config, dataset_raw_dir, resolve_path
from src.utils.provenance import check_gate, read_manifest
from src.utils.logging_utils import get_logger

log = get_logger("env_check")
PKGS = ["numpy", "pandas", "sklearn", "joblib", "yaml", "streamlit", "plotly", "wfdb", "pytest", "pyarrow", "matplotlib", "shap", "requests", "torch"]


def main():
    ok = True
    log.info("Python %s", sys.version.split()[0])
    for p in PKGS:
        try:
            m = importlib.import_module(p)
            log.info("OK   %-10s %s", p, getattr(m, "__version__", "?"))
        except Exception as e:
            ok = False
            log.error("MISSING %s (%s)", p, e)
    cfg = load_config()
    for ds in cfg["datasets"]:
        d = dataset_raw_dir(cfg, ds)
        status = "found" if d.exists() and any(d.iterdir()) else "NOT downloaded yet"
        log.info("dataset %-8s -> %s (%s)", ds, d, status)
    log.info("--- data origin (decides whether models can be shown as available) ---")
    for ds in cfg["datasets"]:
        d = dataset_raw_dir(cfg, ds)
        if d.exists() and any(d.iterdir()):
            m = read_manifest(d); log.info("data %-8s origin=%s (%s)", ds, m.get("data_origin"), m.get("source"))
    log.info("--- model availability gate (artifact + provenance + real data + evaluation report) ---")
    arts, reps = resolve_path(cfg, "artifacts_dir"), resolve_path(cfg, "reports_dir")
    for name, rpt in (("sepsis", "sepsis_metrics.json"), ("forecast", "forecast_metrics.json"), ("ecg", "ecg_metrics.json"),
                      ("diabetes", "diabetes_metrics.json"), ("cardiovascular", "cardiovascular_metrics.json"),
                      ("hypertension", "hypertension_metrics.json")):
        g = check_gate(arts / name, reps / rpt, allow_test=False)
        log.info("model %-9s %-12s %s", name, g["state"].upper(), g["reason"] or f"origin={g['origin']} trained_at={g['trained_at']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
