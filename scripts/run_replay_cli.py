"""Run a full replay in the terminal (no dashboard).  historical data -> twin -> model -> anomaly -> alerts -> SQLite.
  python scripts/run_replay_cli.py                 # first septic patient of the TEST split (real historical data)
  python scripts/run_replay_cli.py --synthetic     # clearly-labelled SYNTHETIC demo patient
  python scripts/run_replay_cli.py --subject sepsis:p000123
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.data_ingestion.sepsis_loader import FEATURES, VITALS
from src.digital_twin.engine import TwinEngine
from src.anomaly_detection.rolling import RollingBaselineDetector
from src.alerts.manager import AlertManager, PROTOTYPE_RULES
from src.database.db import Database
from src.simulation.replay import DataReplayEngine
from src.simulation.builders import build_observations, make_synthetic_observations
from src.models.sepsis_inference import SepsisRiskModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--subject")
    ap.add_argument("--split", default="test"); a = ap.parse_args()
    cfg = load_config()
    art = resolve_path(cfg, "artifacts_dir") / "sepsis"
    try:
        model = SepsisRiskModel(art)
    except FileNotFoundError as e:
        print("WARNING:", e, "-> running without model (anomaly detection only)"); model = None
    if a.synthetic:
        obs, label = make_synthetic_observations(), "SYNTHETIC demo data (not a real patient)"
    else:
        proc = resolve_path(cfg, "processed_dir")
        df = pd.read_parquet(proc / "sepsis_raw_combined.parquet")
        splits = json.loads((proc / "sepsis_splits.json").read_text(encoding="utf-8"))["splits"]
        ids = [a.subject] if a.subject else splits[a.split]
        if not a.subject:
            septic = df[df.subject_id.isin(ids)].groupby("subject_id")["SepsisLabel"].max()
            ids = list(septic[septic == 1].index)
        sid = ids[0]
        obs, label = build_observations(df[df.subject_id == sid]), f"historical replay of {sid} (simulation, not live)"
    db = Database(resolve_path(cfg, "database"))
    am = AlertManager(cfg.get("alerts"), db)
    an = cfg.get("anomaly", {})
    eng = TwinEngine("sepsis", FEATURES, VITALS, model, RollingBaselineDetector(
        an.get("z_threshold", 3.0), an.get("window", 24), an.get("min_points", 6)), am, db)
    sid = obs[0].subject_id
    eng.reset_subject(sid)
    print(f"\n=== {label} ===\n{PROTOTYPE_RULES}\n")

    def cb(o):
        r = eng.process(o)
        rs = r["risk"]["risk_score"] if r["risk"] else float("nan")
        lab = ""
        print(f"hour {o.hour:3d} | quality={r['state']['data_quality']:7s} | risk={rs:5.2f} | status={r['state']['status']:24s}"
              f"| new alerts: {[x['severity'] + ':' + x['type'] for x in r['new_alerts']]}")
        return r
    rp = DataReplayEngine(obs, cb, base_interval=0.0)
    while rp.step() is not None:
        pass
    print("\nAlert history:")
    for al in am.history(sid):
        print(f"  [{al['severity']:8s}] hour {al['hour']:3d} {al['type']:7s} {al['alert_id']} - {al['explanation'][:150]}")
    print("\nSuppressed duplicates:", am.suppressed, "| DB rows:", db.counts())


if __name__ == "__main__":
    main()
