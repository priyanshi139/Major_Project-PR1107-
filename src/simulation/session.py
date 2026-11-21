"""ReplaySession: glue between replay engine, twin engine, model, DB and the dashboard.
Thread-safe snapshots so the UI can read while a background thread replays."""
import json, threading
import pandas as pd
from src.utils.provenance import check_gate
from src.utils.config import resolve_path
from src.data_ingestion.sepsis_loader import FEATURES, VITALS
from src.digital_twin.engine import TwinEngine
from src.anomaly_detection.rolling import RollingBaselineDetector
from src.alerts.manager import AlertManager
from src.database.db import Database
from src.simulation.replay import DataReplayEngine
from src.simulation.builders import build_observations, make_synthetic_observations
from src.models.sepsis_inference import SepsisRiskModel
from src.digital_twin.whatif import compare_scenarios, LIMITS
from src.digital_twin.models import Observation
import re
import numpy as np
from src.forecasting.baselines import ForecastService
from src.ecg.inference import EcgBeatModel


def load_assets(cfg) -> dict:
    """Load processed data, splits and models. Never raises for missing files: reports what is missing.
    A model is loaded for use ONLY if it passes the availability gate (trained artifact + provenance + real public data + matching evaluation
    report). Otherwise assets[name] is None and assets['gates'][name] explains why."""
    proc, arts, reps = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "artifacts_dir"), resolve_path(cfg, "reports_dir")
    out = {"df": None, "splits": None, "model": None, "data_error": None, "model_error": None, "data_provenance": None,
           "forecaster": None, "forecast_error": None, "ecg_model": None, "ecg_error": None, "gates": {}, "tabular": {}, "tabular_errors": {}}
    try:
        pq, sj = proc / "sepsis_raw_combined.parquet", proc / "sepsis_splits.json"
        if not pq.exists() or not sj.exists():
            raise FileNotFoundError("Processed sepsis data not found. Run scripts/prepare_sepsis_data.py")
        out["df"] = pd.read_parquet(pq)
        sj_data = json.loads(sj.read_text(encoding="utf-8"))
        out["splits"] = sj_data["splits"]
        out["data_provenance"] = sj_data.get("provenance") or {"data_origin": "unverified", "source": "splits file has no provenance; re-run prepare_sepsis_data.py"}
    except Exception as e:
        out["data_error"] = str(e)
    specs = [("sepsis", "model", "model_error", lambda: SepsisRiskModel(arts / "sepsis"), arts / "sepsis", reps / "sepsis_metrics.json"),
             ("forecast", "forecaster", "forecast_error", lambda: ForecastService(arts / "forecast"), arts / "forecast", reps / "forecast_metrics.json"),
             ("ecg", "ecg_model", "ecg_error", lambda: EcgBeatModel(arts / "ecg"), arts / "ecg", reps / "ecg_metrics.json")]
    for name, key, err, loader, adir, rpt in specs:
        gate = check_gate(adir, rpt)
        out["gates"][name] = gate
        if not (adir / ("forecasters.joblib" if name == "forecast" else "model.joblib")).exists():
            out[err] = gate["reason"] or f"{name} artifact not found"
            continue
        if not (gate["passed"] or gate.get("test_mode_loadable")):
            out[err] = gate["reason"]
            continue
        try:
            out[key] = loader()
            if not gate["passed"]:
                gate["reason"] = "TEST MODE (HEALTHTWIN_ALLOW_TEST_MODELS=1): " + gate["reason"]
        except Exception as e:
            out[err] = str(e)
    # Each tabular disease has its OWN artifact, report and gate (diabetes / cardiovascular / hypertension).
    from src.diseases.tabular import SPECS as TAB_SPECS, TabularRiskModel
    for k in TAB_SPECS:
        adir, rpt = arts / k, reps / f"{k}_metrics.json"
        gate = check_gate(adir, rpt)
        out["gates"][k], out["tabular"][k] = gate, None
        if not (adir / "model.joblib").exists():
            out["tabular_errors"][k] = gate["reason"] or f"{k} artifact not found"
            continue
        if not (gate["passed"] or gate.get("test_mode_loadable")):
            out["tabular_errors"][k] = gate["reason"]
            continue
        try:
            out["tabular"][k] = TabularRiskModel(adir)
            if not gate["passed"]:
                gate["reason"] = "TEST MODE (HEALTHTWIN_ALLOW_TEST_MODELS=1): " + gate["reason"]
        except Exception as e:
            out["tabular_errors"][k] = str(e)
    return out


class ReplaySession:
    def __init__(self, cfg, assets=None):
        a = assets or {}
        self.cfg, self.df, self.splits, self.model = cfg, a.get("df"), a.get("splits"), a.get("model")
        self.data_error, self.model_error = a.get("data_error"), a.get("model_error")
        self.lock = threading.RLock()
        self.db = Database(resolve_path(cfg, "database"))
        self.alerts = AlertManager(cfg.get("alerts"), self.db)
        an = cfg.get("anomaly", {})
        self.engine = TwinEngine("sepsis", FEATURES, VITALS, self.model, RollingBaselineDetector(
            an.get("z_threshold", 3.0), an.get("window", 24), an.get("min_points", 6)), self.alerts, self.db)
        self.base_interval = cfg.get("replay", {}).get("base_interval_seconds", 1.0)
        self.speed, self.replay, self.subject_id = 1.0, None, None
        self.provenance, self.labels, self._uploaded = None, {}, None
        self._outcomes = (self.df.groupby("subject_id")["SepsisLabel"].max() if self.df is not None else None)

    # ---- subject selection ----
    def subject_choices(self, split="test", only_septic=False):
        if self.df is None:
            return []
        ids = self.splits[split]
        if only_septic:
            ids = [i for i in ids if self._outcomes.get(i, 0) == 1]
        return list(ids)

    def patient_cards(self, n=6, split="test", seed=7):
        """A small, deterministic, mixed selection of real historical patients for the 'choose a patient' cards."""
        if self.df is None:
            return []
        rng = np.random.default_rng(seed)
        ids = self.splits[split]
        pos = [i for i in ids if self.has_sepsis_label(i)]; neg = [i for i in ids if not self.has_sepsis_label(i)]
        pick = list(rng.choice(pos, min(n // 2, len(pos)), replace=False)) + list(rng.choice(neg, min(n - n // 2, len(neg)), replace=False))
        rng.shuffle(pick)
        first = self.df[self.df.subject_id.isin(pick)].groupby("subject_id")
        out = []
        for i in pick:
            g = first.get_group(i)
            age = g["Age"].dropna().iloc[0] if g["Age"].notna().any() else None
            sex = g["Gender"].dropna().iloc[0] if g["Gender"].notna().any() else None
            out.append({"id": i, "age": age, "sex": {0: "Female", 1: "Male"}.get(int(sex)) if sex is not None else None,
                        "hours": int(len(g)), "sepsis_recorded": self.has_sepsis_label(i)})
        return out

    def has_sepsis_label(self, sid):
        return bool(self._outcomes is not None and self._outcomes.get(sid, 0) == 1)

    @property
    def is_synthetic(self):
        return self.provenance == "synthetic"

    @property
    def has_uploaded(self):
        return self._uploaded is not None

    def _install(self, obs, labels, provenance):
        if self.replay:
            self.replay.stop()
        with self.lock:
            if self.subject_id:
                self.engine.reset_subject(self.subject_id)
            self.subject_id, self.provenance, self.labels = obs[0].subject_id, provenance, labels
            self.engine.reset_subject(self.subject_id)
        self.replay = DataReplayEngine(obs, self._on_obs, on_reset=self._on_reset,
                                       base_interval=self.base_interval, speed=self.speed)

    def select_subject(self, sid):
        if self.df is None:
            raise RuntimeError("No processed data loaded")
        p = self.df[self.df.subject_id == sid]
        if p.empty:
            raise KeyError(f"Unknown subject {sid}")
        self._install(build_observations(p), dict(zip(p["hour"], p["SepsisLabel"])), "historical")

    def select_synthetic(self, seed=42):
        self._install(make_synthetic_observations(seed=seed), {}, "synthetic")

    # ---- live entry: readings typed in by the user, each treated as the next hour ----
    def start_live(self, name="patient"):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))[:30].strip("_") or "patient"
        if self.replay:
            self.replay.stop()
        with self.lock:
            if self.subject_id:
                self.engine.reset_subject(self.subject_id)
            self.subject_id, self.provenance, self.labels = f"sepsis:LIVE_{slug}", "live_entry", {}
            self.engine.reset_subject(self.subject_id)
        self.replay = None

    def add_reading(self, values: dict) -> dict:
        if self.provenance != "live_entry":
            raise RuntimeError("Start a live entry session first")
        clean = {}
        for k, v in values.items():
            if v is None:
                continue
            if k in LIMITS and not (LIMITS[k][0] <= float(v) <= LIMITS[k][1]):
                raise ValueError(f"{k}={v} is outside the accepted range {LIMITS[k][0]}-{LIMITS[k][1]}")
            clean[k] = float(v)
        if not any(clean.get(v) is not None for v in VITALS):
            raise ValueError("Enter at least one vital sign")
        with self.lock:
            twin = self.engine.get_twin(self.subject_id)
            hour = 0 if twin.n_observations == 0 else twin.current_hour + 1
            vals = {f: None for f in FEATURES}; vals.update(clean)
            return self.engine.process(Observation(self.subject_id, "sepsis", hour, vals))

    def select_uploaded(self, clean_df):
        """clean_df comes from validate_patient_upload (already validated)."""
        self._uploaded = clean_df
        self.reselect_uploaded()

    def reselect_uploaded(self):
        if self._uploaded is None:
            raise RuntimeError("No uploaded patient")
        d = self._uploaded
        labels = dict(zip(d["hour"], d["SepsisLabel"])) if "SepsisLabel" in d.columns else {}
        self._install(build_observations(d), labels, "uploaded")

    def whatif(self, adjustments, start_hour):
        if self.model is None:
            raise RuntimeError("No trained model loaded; what-if needs the sepsis model")
        with self.lock:
            twin = self.engine.twins.get(self.subject_id)
            hist = twin.history_df() if twin is not None and twin.n_observations else None
        return compare_scenarios(self.model, hist, adjustments, start_hour)

    # ---- callbacks / controls ----
    def _on_obs(self, o):
        with self.lock:
            return self.engine.process(o)

    def _on_reset(self):
        with self.lock:
            if self.subject_id:
                self.engine.reset_subject(self.subject_id)

    def _need(self):
        if self.replay is None:
            raise RuntimeError("Live entry: add readings with the form" if self.provenance == "live_entry"
                               else "Load a patient from the Home page first")
        return self.replay

    def start(self): self._need().start()
    def pause(self): self._need().pause()
    def resume(self): self._need().resume()
    def reset(self): self._need().reset()

    def step(self):
        r = self._need()
        if r.status != "running":
            return r.step()

    def set_speed(self, s):
        self.speed = s
        if self.replay:
            self.replay.set_speed(s)

    @property
    def status(self):
        if self.provenance == "live_entry":
            return "live entry"
        return self.replay.status if self.replay else "no_patient"

    def snapshot(self) -> dict:
        with self.lock:
            twin = self.engine.twins.get(self.subject_id)
            has = twin is not None and twin.n_observations > 0
            cur = twin.current_hour if has else None
            return {
                "status": self.status, "position": self.replay.position if self.replay else (twin.n_observations if has else 0),
                "total": self.replay.total if self.replay else (twin.n_observations if has else 0), "subject_id": self.subject_id,
                "is_synthetic": self.is_synthetic, "provenance": self.provenance, "has_data": has,
                "state": twin.state().to_dict() if has else None,
                "risk_history": list(twin.risk_history) if has else [],
                "history_df": twin.history_df() if has else pd.DataFrame(),
                "snapshots": twin.snapshots if has else [],
                "transitions": list(twin.transitions) if has else [],
                "alerts": [dict(a) for a in self.alerts.history(self.subject_id)],
                "pending_alerts": len(self.alerts.pending(self.subject_id)),
                "suppressed": self.alerts.suppressed,
                "labels_so_far": {h: l for h, l in self.labels.items() if cur is not None and h <= cur},
                "model_name": getattr(self.model, "name", None), "model_error": self.model_error,
                "model_threshold": getattr(self.model, "threshold", None),
                "critical_threshold": self.alerts.critical_threshold,
            }
