"""Train + evaluate each disease model separately on its OWN public dataset.
  python scripts/download_tabular.py
  python scripts/train_tabular.py                 # every disease whose data file exists
  python scripts/train_tabular.py diabetes        # one disease
Outputs: artifacts/<disease>/model.joblib + model_meta.json, reports/<disease>_metrics.json (+ curves png).
A disease whose data file is missing is skipped with a message (never faked)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_global_seed
from src.diseases.tabular import SPECS, RAW_DIR, train_disease


def main():
    cfg = load_config(); seed = set_global_seed(cfg["project"]["seed"])
    keys = sys.argv[1:] or list(SPECS)
    for k in keys:
        if k not in SPECS:
            print(f"unknown disease '{k}'. Choose from: {', '.join(SPECS)}"); continue
        if not (RAW_DIR / SPECS[k].csv).exists():
            print(f"\n{SPECS[k].name}: SKIPPED - {RAW_DIR / SPECS[k].csv} not found (run scripts/download_tabular.py)."); continue
        rep = train_disease(k, resolve_path(cfg, "artifacts_dir") / k, resolve_path(cfg, "reports_dir"), seed)
        t, m = rep["results"]["test"], rep["meta"]
        print(f"\n{SPECS[k].name}: selected={m['selected_model']} (CV AUROC {rep['cv_selection'][m['selected_model']]['cv_auroc_mean']:.3f}) split={m['split_sizes']} data={m['provenance'].get('data_origin')}")
        print(f"  TEST AUROC={t['auroc']:.3f} (95% CI {t['auroc_ci95'][0]:.3f}-{t['auroc_ci95'][1]:.3f})  AUPRC={t['auprc']:.3f} "
              f"(random={t['auprc_random_baseline']:.3f})  P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} Brier={t['brier']:.3f}")
        for w in rep["warnings"]:
            print("  WARNING:", w)


if __name__ == "__main__":
    main()
