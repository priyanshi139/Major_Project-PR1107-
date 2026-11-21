"""Phase 2: load raw sepsis .psv files, summarise, create patient-wise splits.

Usage (PowerShell, from project root):
  python scripts/prepare_sepsis_data.py --zip "C:\\path\\to\\physionet_zip.zip"   # extracts into data/raw/...
  python scripts/prepare_sepsis_data.py                                          # raw folder already filled
  python scripts/prepare_sepsis_data.py --limit 500                              # quick trial on 500 patients
  python scripts/prepare_sepsis_data.py --confirm-physionet-origin               # files were NOT fetched by download_sepsis.py (e.g. an older V4 download or a copied
                                                                                 # folder): you confirm they are the unmodified PhysioNet/CinC 2019 files
Nothing is downloaded from the internet by this script. The data origin (real_public / user_confirmed_real / unverified) is recorded and decides
whether a model trained on it can ever be shown as 'available'.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path, dataset_raw_dir
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed
from src.utils.provenance import read_manifest, write_manifest, sha256_file
from src.data_ingestion.sepsis_loader import extract_zip, load_all
from src.preprocessing.sepsis_preprocessing import patient_wise_split, patient_outcomes

log = get_logger("prepare_sepsis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="Path to a manually downloaded zip to extract into data/raw")
    ap.add_argument("--limit", type=int, default=None, help="Load only first N patient files")
    ap.add_argument("--confirm-physionet-origin", action="store_true",
                    help="I confirm the files in data/raw are the unmodified PhysioNet/CinC 2019 files (records origin 'user_confirmed_real')")
    a = ap.parse_args()

    cfg = load_config()
    seed = set_global_seed(cfg["project"]["seed"])
    raw = dataset_raw_dir(cfg, "sepsis")
    if a.zip:
        log.info("Extracting %s -> %s", a.zip, raw)
        extract_zip(a.zip, raw)
        write_manifest(raw, "user_confirmed_real", "user-supplied archive", archive=Path(a.zip).name, archive_sha256=sha256_file(a.zip))
    elif a.confirm_physionet_origin:
        write_manifest(raw, "user_confirmed_real", "user confirmed unmodified PhysioNet/CinC 2019 files already present in data/raw")
    prov = read_manifest(raw)
    log.info("Data origin: %s (%s)", prov.get("data_origin"), prov.get("source"))
    if prov.get("data_origin") not in ("real_public", "user_confirmed_real"):
        log.warning("Origin is NOT verified: models trained on this data will NOT be shown as available. "
                    "If these are the real PhysioNet files, re-run with --confirm-physionet-origin.")

    df = load_all(raw, limit=a.limit)
    out = patient_outcomes(df)
    log.info("Patients: %d | Rows: %d | Septic patients: %d (%.2f%%)",
             out.size, len(df), int(out.sum()), 100 * out.mean())
    log.info("Row-level positive rate: %.3f%%  (imbalance -> use AUPRC, not accuracy)",
             100 * df["SepsisLabel"].mean())

    sp = cfg["sepsis"]["split"]
    splits = patient_wise_split(df, sp["train_frac"], sp["val_frac"], sp["test_frac"], seed,
                                sp["stratify_by_patient_outcome"])
    proc = resolve_path(cfg, "processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc / "sepsis_raw_combined.parquet", index=False)
    (proc / "sepsis_splits.json").write_text(json.dumps(
        {"seed": seed, "n_patients": int(out.size), "limit": a.limit, "provenance": prov, "splits": splits}, indent=1), encoding="utf-8")
    for k, v in splits.items():
        pos = out.loc[v].mean() if v else float("nan")
        log.info("%-5s patients=%d  septic=%.2f%%", k, len(v), 100 * pos)
    log.info("Saved to %s", proc)


if __name__ == "__main__":
    main()
