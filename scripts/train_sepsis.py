"""Phase 3: train + evaluate sepsis baselines. Run after prepare_sepsis_data.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_global_seed
from src.models.train_sepsis import run_training


def main():
    cfg = load_config(); seed = set_global_seed(cfg["project"]["seed"])
    proc = resolve_path(cfg, "processed_dir")
    pq, sj = proc / "sepsis_raw_combined.parquet", proc / "sepsis_splits.json"
    if not pq.exists() or not sj.exists():
        sys.exit("Processed data not found. Run: python scripts/prepare_sepsis_data.py")
    df = pd.read_parquet(pq)
    sj_data = json.loads(sj.read_text(encoding="utf-8"))
    prov = sj_data.get("provenance") or {"data_origin": "unverified", "source": "splits file has no provenance; re-run prepare_sepsis_data.py"}
    print("Training data origin:", prov.get("data_origin"), "-", prov.get("source"))
    run_training(df, sj_data["splits"], resolve_path(cfg, "artifacts_dir") / "sepsis", resolve_path(cfg, "reports_dir"), seed, provenance=prov)


if __name__ == "__main__":
    main()
