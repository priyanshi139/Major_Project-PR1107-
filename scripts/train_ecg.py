"""Train and evaluate the ECG beat classifier (record-wise DS1/DS2 split). Run after download_ecg.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path, dataset_raw_dir
from src.utils.seed import set_global_seed
from src.ecg.train import run_ecg_training


def main():
    cfg = load_config(); seed = set_global_seed(cfg["project"]["seed"])
    rep = run_ecg_training(dataset_raw_dir(cfg, "ecg"), resolve_path(cfg, "artifacts_dir") / "ecg",
                           resolve_path(cfg, "reports_dir"), seed)
    print("Splits (records):", rep["meta"]["splits"])
    for n, r in rep["results"].items():
        t = r["test"]
        print(f"{n:24s} TEST macro-F1={t['macro_f1_present_classes']:.3f} balanced-acc={t['balanced_accuracy']:.3f} "
              + " ".join(f"{c}: R={v['recall']:.2f}/P={v['precision']:.2f}(n={v['support']})" for c, v in t['per_class'].items()))
    print("Selected:", rep["meta"]["selected_model"])


if __name__ == "__main__":
    main()
