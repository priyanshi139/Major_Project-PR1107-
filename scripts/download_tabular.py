"""Download the public datasets used by diabetes, cardiovascular and hypertension models.

Sources are public GitHub mirrors, not original dataset distributions. Each file's SHA-256
and row/positive counts are recorded in PROVENANCE.json.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.provenance import sha256_file, write_manifest, read_manifest, MANIFEST

SOURCES = {
    "pima_diabetes.csv": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "framingham.csv": "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-existing", action="store_true",
                    help="record CSV files you placed yourself as user-confirmed public datasets")
    args = ap.parse_args()
    dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "public_tabular"
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    if args.confirm_existing:
        for f in dest.glob("*.csv"):
            if f.name in SOURCES:
                files[f.name] = {"sha256": sha256_file(f), "how": "user placed this file and confirmed its public-dataset origin"}
        origin = "user_confirmed_real"
    else:
        import requests
        for name, url in SOURCES.items():
            out = dest / name
            r = requests.get(url, timeout=60, headers={"User-Agent": "HealthTwin-academic-project"})
            r.raise_for_status()
            content = r.content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            out.write_text("\n".join(line for line in content.split("\n") if line.strip()) + "\n", encoding="utf-8")
            files[name] = {"sha256": sha256_file(out), "url": url}
            print("fetched", name)
        origin = "real_public"
        old = read_manifest(dest)
        for name, meta in (old.get("files") or {}).items():
            if name in SOURCES and name not in files and (dest / name).exists() and sha256_file(dest / name) == meta.get("sha256"):
                files[name] = meta
    write_manifest(dest, origin, "public GitHub mirrors (see per-file URLs)", files=files)
    for name, meta in files.items():
        print(f"  {name}  sha256 {meta['sha256'][:16]}")
    print(f"Provenance written to {dest / MANIFEST}")


if __name__ == "__main__":
    main()
