"""Download PhysioNet/CinC Challenge 2019 .psv files (open-access dataset) into data/raw.

Base URL (from the PhysioNet 'Access the files' section):
  https://physionet.org/files/challenge-2019/1.0.0/

Usage (PowerShell, from project root):
  python scripts/download_sepsis.py --limit 500            # first 500 patients per set (trial)
  python scripts/download_sepsis.py --all                  # full download (~40k small files, takes a while)
  python scripts/download_sepsis.py --sets A --limit 2000  # only training_setA
  python scripts/download_sepsis.py --limit 3000 --workers 8   # parallel (much faster)
Already-downloaded files are skipped, so you can stop (Ctrl+C) and re-run to resume.
Please read the dataset license/terms on the PhysioNet page before downloading.
"""
import argparse, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from src.utils.config import load_config, dataset_raw_dir
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger("download_sepsis")
BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
SETS = {"A": "training_setA", "B": "training_setB"}


def parse_listing(html: str) -> list:
    """Return sorted unique .psv file names found in a PhysioNet directory listing."""
    names = re.findall(r'href="([^"?#]+\.psv)"', html)
    return sorted({n.split("/")[-1] for n in names})


def fetch_listing(session, set_dir: str) -> list:
    url = urljoin(BASE, set_dir + "/")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    names = parse_listing(r.text)
    if not names:
        raise RuntimeError(f"No .psv files found in listing at {url}")
    return names


def download_file(session, url: str, dest: Path, retries=3) -> bool:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, dest.name, e)
            time.sleep(2 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--limit", type=int, help="Max files per set (trial download)")
    g.add_argument("--all", action="store_true", help="Download every file")
    ap.add_argument("--sets", nargs="+", default=["A", "B"], choices=list(SETS))
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds to wait after each file (per worker)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel downloads (try 4-8; be polite to the server)")
    a = ap.parse_args()

    cfg = load_config()
    raw = dataset_raw_dir(cfg, "sepsis")
    local = threading.local()

    def get_session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers["User-Agent"] = "HealthTwin-academic-project"
        return local.s

    def task(set_dir, out_dir, n):
        dest = out_dir / n
        if dest.exists() and dest.stat().st_size > 0:
            return "skipped"
        ok = download_file(get_session(), urljoin(BASE, f"{set_dir}/{n}"), dest)
        if a.delay:
            time.sleep(a.delay)
        return "done" if ok else "failed"

    failed = 0
    for s_key in a.sets:
        set_dir = SETS[s_key]
        out_dir = raw / set_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        names = fetch_listing(get_session(), set_dir)
        log.info("%s: %d files available on server", set_dir, len(names))
        if a.limit:
            names = names[: a.limit]
        counts = {"done": 0, "skipped": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = [ex.submit(task, set_dir, out_dir, n) for n in names]
            for i, f in enumerate(as_completed(futs), 1):
                counts[f.result()] += 1
                if i % 500 == 0:
                    log.info("%s: %d/%d processed", set_dir, i, len(names))
        failed += counts["failed"]
        log.info("%s: downloaded=%d skipped(existing)=%d failed=%d",
                 set_dir, counts["done"], counts["skipped"], counts["failed"])
    if failed:
        log.error("%d files failed. Re-run the same command to retry them.", failed)
        sys.exit(1)
    counts_now = {k: len(list((raw / d).glob("*.psv"))) for k, d in SETS.items() if (raw / d).exists()}
    write_manifest(raw, "real_public", BASE, dataset="PhysioNet/CinC Challenge 2019", files_present=counts_now,
                   note="Files fetched by scripts/download_sepsis.py from physionet.org")
    log.info("Done. Files are in %s (provenance recorded in PROVENANCE.json)", raw)


if __name__ == "__main__":
    main()
