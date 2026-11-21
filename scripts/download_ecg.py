"""Download MIT-BIH Arrhythmia Database records with the wfdb package (public PhysioNet data; read its license page).
  python scripts/download_ecg.py              # all 48 records (~70-100 MB)
  python scripts/download_ecg.py --records 100 101 200
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, dataset_raw_dir
from src.utils.provenance import write_manifest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", nargs="+"); a = ap.parse_args()
    import wfdb
    dest = dataset_raw_dir(load_config(), "ecg"); dest.mkdir(parents=True, exist_ok=True)
    print("Downloading MIT-BIH (mitdb) to", dest)
    wfdb.dl_database("mitdb", dl_dir=str(dest), records=a.records or "all")
    n = len(list(dest.glob("*.hea")))
    write_manifest(dest, "real_public", "https://physionet.org/content/mitdb/1.0.0/", dataset="MIT-BIH Arrhythmia Database", records_present=n)
    print("Done:", n, "records (provenance recorded in PROVENANCE.json)")


if __name__ == "__main__":
    main()
