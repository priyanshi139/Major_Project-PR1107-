"""Train short-horizon vital-sign forecasters and compare them with persistence. Run after prepare_sepsis_data.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_global_seed
from src.forecasting.baselines import train_forecasters, save_forecasters


def main():
    cfg = load_config(); set_global_seed(cfg["project"]["seed"])
    proc = resolve_path(cfg, "processed_dir")
    if not (proc / "sepsis_raw_combined.parquet").exists():
        sys.exit("Run: python scripts/prepare_sepsis_data.py")
    df = pd.read_parquet(proc / "sepsis_raw_combined.parquet")
    sj = json.loads((proc / "sepsis_splits.json").read_text(encoding="utf-8"))
    models, metrics = train_forecasters(df, sj["splits"])
    save_forecasters(models, metrics, resolve_path(cfg, "artifacts_dir") / "forecast", resolve_path(cfg, "reports_dir"), provenance=sj.get("provenance"))
    print(f"{'var':6s} {'h':>2s} {'enabled':>8s} {'test MAE model':>15s} {'test MAE persist':>17s}")
    for r in metrics.values():
        if "test_mae_model" in r:
            print(f"{r['variable']:6s} {r['horizon']:2d} {str(r['enabled']):>8s} {r['test_mae_model']:15.3f} {r['test_mae_persistence']:17.3f}")
        else:
            print(f"{r['variable']:6s} {r['horizon']:2d} {'False':>8s}  {r['note']}")


if __name__ == "__main__":
    main()
