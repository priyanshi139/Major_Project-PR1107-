"""PatientService: the patient-centric workflow.
registration -> profile -> twin -> observations (manual / simulated live / uploaded / historical) -> twin update -> prediction ->
anomaly + alerts -> timeline events -> what-if / forecast / insights. Reuses TwinEngine, AlertManager, DataReplayEngine, the trained model."""
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.utils.config import resolve_path
from src.data_ingestion.sepsis_loader import FEATURES, VITALS
from src.data_ingestion.schema import validate_values, SPECS
from src.database.patient_store import PatientStore
from src.digital_twin.engine import TwinEngine
from src.digital_twin.models import Observation
from src.digital_twin.whatif import compare_scenarios, scenario_state
from src.anomaly_detection.rolling import RollingBaselineDetector
from src.alerts.manager import AlertManager
from src.simulation.replay import DataReplayEngine
from src.simulation.feed import generate_feed, SCENARIOS, BASE
from src.models.explain import occlusion_explanation
from src.insights.evidence import generate_insights
from src.diseases.registry import build_registry

CONDITIONS = ["Sepsis risk monitoring", "Type 2 diabetes", "Cardiovascular disease", "Hypertension", "Other"]
GENDERS = ["Female", "Male", "Other", "Unknown"]
ICU_UNITS = ["Unknown", "MICU", "SICU"]
SOURCE_LABELS = {"manual_entry": "Manual entry", "simulated_live": "SIMULATED LIVE DATA", "synthetic_demo": "SYNTHETIC demo data",
                 "uploaded": "Uploaded file", "historical_dataset": "Public dataset record", "historical_replay": "Public dataset record (replayed)",
                 "unspecified": "Unspecified"}
SYNTHETIC_SOURCES = {"simulated_live", "synthetic_demo"}


def _now():
    return datetime.now().replace(microsecond=0)


def validate_profile(p: dict) -> dict:
    errs = []
    name = str(p.get("name") or "").strip() or "Demo patient"
    if len(name) > 60: errs.append("Name/identifier must be at most 60 characters")
    try:
        age = int(p.get("age")); assert 0 <= age <= 120
    except Exception:
        age = None; errs.append("Age must be a whole number between 0 and 120")
    gender = p.get("gender") or "Unknown"
    if gender not in GENDERS: errs.append(f"Gender must be one of {GENDERS}")
    conds = list(p.get("conditions") or [])
    if any(c not in CONDITIONS for c in conds): errs.append(f"Conditions must be from {CONDITIONS}")
    history = str(p.get("history") or "")
    if len(history) > 500: errs.append("Medical history must be at most 500 characters")
    time_basis = p.get("time_basis") or "clock"
    ms = p.get("monitoring_start")
    if time_basis == "clock":
        try:
            ms = (datetime.fromisoformat(str(ms)) if ms else _now().replace(minute=0, second=0)).isoformat()
        except ValueError:
            errs.append("Monitoring start is not a valid date/time")
    base, e2 = validate_values({k: v for k, v in (p.get("baseline") or {}).items() if v not in (None, "")})
    errs += e2
    unit = p.get("icu_unit") or "Unknown"
    if unit not in ICU_UNITS: errs.append(f"ICU unit must be one of {ICU_UNITS}")
    if errs:
        raise ValueError("; ".join(errs))
    return {"name": name, "age": age, "gender": gender, "conditions": conds, "history": history, "monitoring_start": ms if time_basis == "clock" else None,
            "time_basis": time_basis, "baseline": base, "icu_unit": unit, "synthetic": bool(p.get("synthetic", True)),
            "origin": p.get("origin") or "manual_registration"}


class PatientService:
    def __init__(self, cfg, assets=None, db_path=None):
        a = assets or {}
        self.cfg, self.assets = cfg, a
        self.model, self.forecaster, self.df, self.splits = a.get("model"), a.get("forecaster"), a.get("df"), a.get("splits")
        self.store = PatientStore(db_path or (resolve_path(cfg, "artifacts_dir") / "patients.db"))
        self.alerts = AlertManager(cfg.get("alerts"), self.store)
        an = cfg.get("anomaly", {})
        self.engine = TwinEngine("sepsis", FEATURES, VITALS, self.model, RollingBaselineDetector(
            an.get("z_threshold", 3.0), an.get("window", 24), an.get("min_points", 6)), self.alerts, self.store)
        self.registry = build_registry(a, resolve_path(cfg, "artifacts_dir"))
        self.tabular_models = a.get("tabular") or {}
        self.lock = threading.RLock()
        self.feeds, self.replay_queues, self._restored, self._xcache = {}, {}, set(), {}

    # ---------- helpers ----------
    @staticmethod
    def subject_id(pid): return f"sepsis:{pid}"

    def risk_category(self, score, threshold):
        if score is None or threshold is None: return None
        crit = self.alerts.critical_threshold
        return "Critical" if score >= crit else "High" if score >= threshold else "Elevated" if score >= 0.5 * threshold else "Low"

    def _timeline(self, pid, etype, title, detail=None, hour=None, ts=None, sev=None, source=None):
        self.store.add_event(pid, etype, title, detail, hour, ts, sev, source)

    # ---------- patients ----------
    def register_patient(self, profile: dict) -> str:
        clean = validate_profile(profile)
        pid = self.store.create_patient(clean)
        self._timeline(pid, "registration", f"Patient registered ({'SYNTHETIC/demo' if clean['synthetic'] else 'non-synthetic'})",
                       {"name": clean["name"], "conditions": clean["conditions"], "origin": clean["origin"]}, ts=_now().isoformat())
        self.engine.get_twin(self.subject_id(pid))
        return pid

    def get_patient(self, pid): return self.store.get_patient(pid)

    def update_patient(self, pid, changes: dict):
        cur = self.store.get_patient(pid)
        if cur is None: raise KeyError(f"Unknown patient {pid}")
        merged = {k: cur.get(k) for k in ("name", "age", "gender", "conditions", "history", "monitoring_start", "time_basis", "baseline", "icu_unit", "synthetic", "origin")}
        has_obs = self.ensure_twin(pid).n_observations > 0
        if has_obs and "monitoring_start" in changes and changes["monitoring_start"] != merged["monitoring_start"]:
            raise ValueError("Monitoring start cannot change after observations exist (it defines the observation hours)")
        merged.update(changes)
        clean = validate_profile(merged)
        self.store.update_patient(pid, clean)
        self._timeline(pid, "profile", "Profile edited", {k: v for k, v in changes.items() if k != "history"}, ts=_now().isoformat())
        return clean

    def list_patients(self, query=None):
        out = self.store.list_patients()
        if query:
            q = query.lower()
            out = [p for p in out if q in p["patient_id"].lower() or q in p["name"].lower() or any(q in c.lower() for c in p["conditions"])]
        return out

    def delete_patient(self, pid):
        self.stop_feed(pid); self.store.delete_patient(pid)
        self.engine.twins.pop(self.subject_id(pid), None); self.alerts.reset_subject(self.subject_id(pid))
        self.replay_queues.pop(pid, None); self._restored.discard(pid)

    # ---------- twin ----------
    def ensure_twin(self, pid):
        """Return the patient's twin; rebuild it from the database after an app restart."""
        sid = self.subject_id(pid)
        with self.lock:
            twin = self.engine.get_twin(sid)
            if twin.n_observations > 0 or pid in self._restored:
                return twin
            self._restored.add(pid)
            rows, meta = self.store.get_observations(sid), self.store.get_obs_meta(sid)
            risks = {r["hour"]: r for r in self.store.get_risks(sid)}
            for r in rows:
                m = meta.get(r["hour"], {})
                obs = Observation(sid, "sepsis", r["hour"], r["values"], m.get("timestamp"), m.get("source") or "unspecified")
                twin.add_observation(obs)
                twin.record_deviations(self.engine.detector.detect(twin, obs))
                if r["hour"] in risks:
                    twin.record_risk({k: v for k, v in risks[r["hour"]].items() if k != "hour"})
            for item in self.store.get_disease_assessments(pid):
                disease = item.get("disease")
                risk = {k: v for k, v in item.items() if k not in {"assessment_id", "disease", "wall_time"}}
                twin.record_disease_risk(disease, risk, hour=item.get("hour"))
            self.alerts.restore(self.store.get_alerts(sid))
            return twin

    def _profile_features(self, profile, hour):
        f = {"ICULOS": float(hour + 1)}
        if profile.get("age") is not None: f["Age"] = float(profile["age"])
        g = {"Female": 0.0, "Male": 1.0}.get(profile.get("gender"))
        if g is not None: f["Gender"] = g
        if profile.get("icu_unit") == "MICU": f.update(Unit1=1.0, Unit2=0.0)
        if profile.get("icu_unit") == "SICU": f.update(Unit1=0.0, Unit2=1.0)
        return f

    def add_observation(self, pid, values: dict, timestamp=None, source="manual_entry") -> dict:
        """Validated observation entry. Blank values stay missing (handled by the existing preprocessing, never filled here)."""
        profile = self.store.get_patient(pid)
        if profile is None: raise KeyError(f"Unknown patient {pid}")
        clean, errs = validate_values(values)
        if errs: raise ValueError("; ".join(errs))
        if not any(v in clean for v in VITALS): raise ValueError("Enter at least one vital sign")
        twin = self.ensure_twin(pid); recs = twin.observation_records()
        ts = timestamp if isinstance(timestamp, datetime) else (datetime.fromisoformat(timestamp) if timestamp else _now())
        last = recs[-1] if recs else None
        if last and last["timestamp"] and ts <= datetime.fromisoformat(last["timestamp"]):
            raise ValueError(f"Timestamp must be later than the last observation ({last['timestamp']})")
        last_hour = last["hour"] if last else -1
        if profile["time_basis"] == "clock":
            hour = int((ts - datetime.fromisoformat(profile["monitoring_start"])).total_seconds() // 3600)
            if hour < 0: raise ValueError("Timestamp is before the monitoring start")
        else:
            hour = last_hour + 1
        slot_note = None
        if hour <= last_hour:
            hour = last_hour + 1
            slot_note = f"Entries less than an hour apart share the model's hourly resolution; this one was placed in hour slot {hour}."
        vals = {f: None for f in FEATURES}; vals.update(self._profile_features(profile, hour)); vals.update(clean)
        res = self._process(pid, Observation(self.subject_id(pid), "sepsis", hour, vals, ts.isoformat(), source))
        res["slot_note"] = slot_note
        return res

    def _process(self, pid, obs) -> dict:
        with self.lock:
            res = self.engine.process(obs)
            twin = self.engine.twins[obs.subject_id]
        risk = res["risk"]
        if risk is not None:
            risk["risk_category"] = self.risk_category(risk["risk_score"], risk["threshold"])
            risk["predicted_at"] = _now().isoformat(); risk["data_timestamp"] = obs.timestamp
            self.store.save_risk(obs.subject_id, obs.module, obs.hour, risk)
        for a in res["new_alerts"]:
            if a["type"] in ("risk", "risk_rise") and self.model is not None:
                try:
                    x = self.explain(pid, force=True)
                    top = [{"feature": i["feature"], "effect": i["effect"]} for i in x["items"][:3]]
                    self.alerts.annotate(a["alert_id"], factors=top, explanation=a["explanation"] + " Largest effects on this score (occlusion): " +
                                         ", ".join(f"{i['feature']} {i['effect']:+.2f}" for i in top) + ".")
                except Exception:
                    pass
        h, ts, src = obs.hour, obs.timestamp, obs.source
        meas = {k: v for k, v in obs.values.items() if v is not None and k in SPECS}
        ev = [("observation", f"Observation recorded - {SOURCE_LABELS.get(src, src)}", {"values": meas, "synthetic": src in SYNTHETIC_SOURCES}, None)]
        if twin.transitions and twin.transitions[-1]["hour"] == h:
            t = twin.transitions[-1]; ev.append(("state_change", f"Twin status: {t['from']} -> {t['to']}".replace("_", " "), t, None))
        if risk:
            ev.append(("prediction", f"Sepsis model score {risk['risk_score']:.2f} ({risk['risk_category']})",
                       {"score": risk["risk_score"], "threshold": risk["threshold"], "model": risk["model_name"], "n_obs": risk["n_observations_used"]}, None))
        for d in res["anomalies"]:
            ev.append(("anomaly", d["explanation"][:140], {"variable": d["variable"], "z": d["z"]}, "info"))
        for a in res["new_alerts"]:
            ev.append(("alert", f"{a['severity'].upper()} alert ({a['type']}): {a['trigger']}", {"alert_id": a["alert_id"], "score": a["score"]}, a["severity"]))
        for t, title, detail, sev in ev:
            self._timeline(pid, t, title, detail, h, ts, sev, src)
        res["hour"], res["timestamp"], res["source"] = h, ts, src
        return res

    def assess_disease(self, pid: str, disease: str, values: dict) -> dict:
        """Run a disease's own tabular model and attach the assessment to this patient's twin.

        These are point-in-time screening/risk estimates, not temporal forecasts. Inputs are
        explicitly supplied for this assessment; missing fields are handled by that model's
        training-set imputation and are disclosed in its response.
        """
        profile = self.store.get_patient(pid)
        if profile is None:
            raise KeyError(f"Unknown patient {pid}")
        if disease not in ("diabetes", "cardiovascular", "hypertension"):
            raise ValueError("Unsupported disease. Choose diabetes, cardiovascular or hypertension.")
        module = self.registry.get(disease)
        if module is None or module.status != "available":
            reason = module.reason if module else "Disease model is not registered"
            raise RuntimeError(f"{disease} model is not validated and available: {reason}")
        model = self.tabular_models.get(disease)
        if model is None:
            raise RuntimeError(f"{disease} model artifact is not loaded")
        # Patient profile supplies only defensible demographics; clinical measurements must be entered.
        payload = dict(values or {})
        if profile.get("age") is not None:
            if payload.get("Age") in (None, ""):
                payload["Age"] = profile["age"]
            if payload.get("age") in (None, ""):
                payload["age"] = profile["age"]
        gender = profile.get("gender")
        if gender in ("Female", "Male"):
            payload.setdefault("male", 1 if gender == "Male" else 0)
        risk = model.predict(payload)
        twin = self.ensure_twin(pid)
        hour = twin.current_hour if twin.n_observations else -1
        risk.update({"risk_category": risk.get("category"), "assessment_type": "cross_sectional",
                     "input_source": "patient_assessment_form", "patient_id": pid,
                     "input_values": {k: v for k, v in payload.items() if v not in (None, "")},
                     "assessed_at": _now().isoformat(), "data_hour": hour, "model_status": module.status,
                     "clinical_notice": "Academic screening estimate only; not a diagnosis or treatment recommendation."})
        if disease == "diabetes" and profile.get("gender") == "Male":
            risk["population_warning"] = "The underlying Pima dataset contains women only; this estimate is outside the dataset's represented population."
        self.store.save_disease_assessment(pid, disease, hour, risk)
        twin.record_disease_risk(disease, risk, hour=hour)
        self._timeline(pid, "disease_assessment",
                       f"{module.name} assessment: {risk['risk_score']:.2f}",
                       {"disease": disease, "risk_score": risk["risk_score"], "threshold": risk.get("threshold"),
                        "n_inputs_given": risk.get("n_inputs_given"), "missing_inputs": risk.get("missing_inputs", [])},
                       hour if hour >= 0 else None, _now().isoformat(), "info", "patient_assessment_form")
        return {"disease": disease, **risk, "assessment_id": self.store.get_disease_assessments(pid, disease)[-1]["assessment_id"]}

    def disease_risk_history(self, pid: str, disease=None) -> dict:
        if self.store.get_patient(pid) is None:
            raise KeyError(f"Unknown patient {pid}")
        self.ensure_twin(pid)
        rows = self.store.get_disease_assessments(pid, disease)
        if disease:
            return {disease: rows}
        return {k: self.store.get_disease_assessments(pid, k) for k in ("diabetes", "cardiovascular", "hypertension")}

    def snapshot(self, pid) -> dict:
        twin = self.ensure_twin(pid); profile = self.store.get_patient(pid); sid = self.subject_id(pid)
        with self.lock:
            has = twin.n_observations > 0
            st = twin.state().to_dict() if has else None
            thr = getattr(self.model, "threshold", None)
            return {"patient": profile, "patient_id": pid, "has_data": has, "state": st,
                    "risk_history": list(twin.risk_history) if has else [],
                    "disease_risk_history": {k: list(v) for k, v in twin.disease_risk_history.items()},
                    "health_trajectory": twin.health_trajectory() if has else {
                        "available": False, "reason": "No observations are available.",
                        "method": "descriptive_linear_trend", "clinical_forecast": False},
                    "baseline_quality": {
                        "status": "established" if has and any(v.get("n_baseline", 0) >= 3 for v in twin.baseline().values()) else ("building" if has else "not_started"),
                        "minimum_measurements_for_variable": 3,
                        "note": "Baseline is calculated from this patient's earliest available observed readings; it is not a population reference."
                    },
                    "history_df": twin.history_df() if has else pd.DataFrame(),
                    "snapshots": twin.snapshots if has else [], "transitions": list(twin.transitions), "obs_records": twin.observation_records(),
                    "alerts": [dict(a) for a in self.alerts.history(sid)], "pending_alerts": len(self.alerts.pending(sid)),
                    "suppressed": self.alerts.suppressed, "model_name": getattr(self.model, "name", None), "model_threshold": thr,
                    "critical_threshold": self.alerts.critical_threshold,
                    "category": self.risk_category(st["risk"].get("risk_score"), st["risk"].get("threshold")) if has and st["risk"] else None}

    def summaries(self) -> list:
        out = []
        for p in self.list_patients():
            sn = self.snapshot(p["patient_id"]); st = sn["state"]; r = st["risk"] if st else {}
            recs = sn["obs_records"]
            out.append({**p, "n_obs": len(recs), "status": st["status"] if st else "no data", "risk_score": r.get("risk_score") if r else None,
                        "category": sn["category"], "flagged": bool(r.get("flagged")) if r else False, "pending_alerts": sn["pending_alerts"],
                        "last_update": (recs[-1]["timestamp"] or f"hour {recs[-1]['hour']}") if recs else None,
                        "sources": sorted({x["source"] for x in recs})})
        return out

    # ---------- explanation / insights / forecast / what-if ----------
    def explain(self, pid, force=False):
        if self.model is None: return None
        twin = self.ensure_twin(pid)
        if twin.n_observations == 0: return None
        key = (pid, twin.current_hour)
        if force or key not in self._xcache:
            self._xcache[key] = occlusion_explanation(self.model, twin.history_df(), top_k=6)
        return self._xcache[key]

    def insights(self, pid, with_occlusion=True):
        sn = self.snapshot(pid)
        return generate_insights(sn, sn["patient"], self.explain(pid) if with_occlusion and sn["has_data"] else None, sn["category"])

    def forecast(self, pid, var, horizon):
        if self.forecaster is None: return {"available": False, "reason": "forecast models not loaded (run scripts/train_forecast.py)"}
        twin = self.ensure_twin(pid)
        return self.forecaster.predict(twin.history_df(), var, horizon)

    def whatif(self, pid, changes, last_k=1) -> dict:
        twin = self.ensure_twin(pid)
        res = scenario_state(self.model, twin.history_df() if twin.n_observations else None, changes, last_k)
        self.store.save_whatif(pid, {"kind": "state", "changes": res["changes"], "applied_to_last": last_k, "hour": res["hour"],
                                     "current_score": res["current"]["risk_score"] if res["current"] else None,
                                     "scenario_score": res["scenario"]["risk_score"] if res["scenario"] else None, "simulated": True})
        if res["changes"]:
            self._timeline(pid, "whatif", "What-if scenario (SIMULATED): " + ", ".join(f"{k} -> {v:g}" for k, v in res["changes"].items()),
                           {"current_score": res["current"]["risk_score"] if res["current"] else None,
                            "scenario_score": res["scenario"]["risk_score"] if res["scenario"] else None}, res["hour"], None, None, "simulated_whatif")
        return res

    def whatif_trajectory(self, pid, adjustments, start_hour):
        twin = self.ensure_twin(pid)
        return compare_scenarios(self.model, twin.history_df() if twin.n_observations else None, adjustments, start_hour)

    # ---------- alerts ----------
    def alert_action(self, alert_id, action, note="", by="dashboard_user"):
        fn = {"acknowledge": self.alerts.acknowledge, "escalate": self.alerts.escalate, "resolve": self.alerts.resolve}[action]
        a = fn(alert_id, by=by, note=note)
        self._timeline(a["patient_id"], "alert_action", f"Alert {alert_id} {a['status']} by {by}" + (f": {note}" if note else ""),
                       {"alert_id": alert_id, "status": a["status"]}, a["hour"], None, None, "human_review")
        return a

    def all_alerts(self):
        for p in self.list_patients(): self.ensure_twin(p["patient_id"])
        return [dict(a) for a in self.alerts.history()]

    # ---------- bulk / data management ----------
    def append_uploaded(self, pid, clean_df) -> int:
        profile = self.store.get_patient(pid); twin = self.ensure_twin(pid); n = 0
        last_h = twin.current_hour if twin.n_observations else -1
        if clean_df["hour"].iloc[0] <= last_h: raise ValueError(f"Uploaded hours must start after the patient's last hour ({last_h})")
        for _, r in clean_df.iterrows():
            vals = {f: (None if (f not in clean_df.columns or pd.isna(r[f])) else float(r[f])) for f in FEATURES}
            vals.update({k: v for k, v in self._profile_features(profile, int(r["hour"])).items() if vals.get(k) is None})
            ts = (datetime.fromisoformat(profile["monitoring_start"]) + timedelta(hours=int(r["hour"]))).isoformat() if profile["time_basis"] == "clock" else None
            self._process(pid, Observation(self.subject_id(pid), "sepsis", int(r["hour"]), vals, ts, "uploaded")); n += 1
        self._timeline(pid, "import", f"Uploaded file: {n} hourly observations added", {"rows": n}, int(clean_df["hour"].iloc[-1]), None, None, "uploaded")
        return n

    def import_historical(self, source_id, n_initial=None) -> str:
        """Register a public-dataset record as a patient. The first n_initial hours are loaded now; the rest can be replayed live."""
        from src.simulation.builders import build_observations
        if self.df is None: raise RuntimeError("Public dataset not prepared (run scripts/prepare_sepsis_data.py)")
        p = self.df[self.df.subject_id == source_id]
        if p.empty: raise KeyError(f"Unknown record {source_id}")
        obs = build_observations(p, "historical_dataset")
        age = p["Age"].dropna(); sex = p["Gender"].dropna()
        pid = self.register_patient({"name": f"PhysioNet record {source_id.split(':')[-1]}", "age": int(age.iloc[0]) if len(age) else 0,
                                     "gender": {0: "Female", 1: "Male"}.get(int(sex.iloc[0])) if len(sex) else "Unknown",
                                     "conditions": ["Sepsis risk monitoring"], "synthetic": False, "origin": "physionet_2019_record",
                                     "time_basis": "relative", "history": "Public de-identified ICU record; no real clock times."})
        n = len(obs) if n_initial is None else max(1, min(n_initial, len(obs)))
        for o in obs[:n]:
            self._process(pid, Observation(self.subject_id(pid), "sepsis", o.hour, o.values, None, "historical_dataset"))
        if n < len(obs): self.replay_queues[pid] = obs[n:]
        return pid

    def create_demo_patients(self, seed=11) -> list:
        """Three SYNTHETIC demo patients with 24 hours of generated history (source 'synthetic_demo')."""
        specs = [("Demo A - stable", 58, "Female", ["Hypertension"], "stable", 0), ("Demo B - deteriorating", 67, "Male", ["Sepsis risk monitoring", "Type 2 diabetes"], "deterioration", 14),
                 ("Demo C - recovering", 45, "Female", [], "recovery", 8)]
        out = []
        for i, (name, age, g, conds, scen, onset) in enumerate(specs):
            start = (_now() - timedelta(hours=24)).replace(minute=0, second=0)
            pid = self.register_patient({"name": name, "age": age, "gender": g, "conditions": conds, "synthetic": True, "origin": "demo_generator",
                                         "monitoring_start": start.isoformat(), "history": "Synthetic demo patient generated by code."})
            st0 = {"HR": 96, "Resp": 21, "Temp": 37.6, "O2Sat": 94, "SBP": 105, "MAP": 72, "DBP": 58} if scen == "recovery" else None
            for it in generate_feed(scen, 24, st0, seed + i, onset):
                self.add_observation(pid, it.values, start + timedelta(hours=it.hour), "synthetic_demo")
            out.append(pid)
        return out

    # ---------- simulated live monitoring ----------
    def _next_timestamp(self, pid):
        recs = self.ensure_twin(pid).observation_records(); prof = self.store.get_patient(pid)
        if recs and recs[-1]["timestamp"]: return datetime.fromisoformat(recs[-1]["timestamp"]) + timedelta(hours=1)
        if prof["time_basis"] == "clock" and not recs: return datetime.fromisoformat(prof["monitoring_start"])
        return _now()

    def start_simulated_feed(self, pid, scenario="stable", n_hours=12, interval=1.0, seed=7):
        self.stop_feed(pid)
        lv = self.ensure_twin(pid).state().latest_values if self.ensure_twin(pid).n_observations else {}
        items = generate_feed(scenario, n_hours, {k: lv[k] for k in BASE if k in lv}, seed)
        rp = DataReplayEngine(items, lambda it: self.add_observation(pid, it.values, self._next_timestamp(pid), "simulated_live"), base_interval=interval)
        self.feeds[pid] = {"replay": rp, "kind": "simulated", "label": "Simulated Live Data", "scenario": scenario}
        rp.start()

    def start_historical_replay(self, pid, interval=1.0):
        q = self.replay_queues.get(pid)
        if not q: raise ValueError("This patient has no remaining historical hours to replay")
        self.stop_feed(pid)
        rp = DataReplayEngine(q, lambda o: self._process(pid, Observation(self.subject_id(pid), "sepsis", o.hour, o.values, None, "historical_replay")),
                              base_interval=interval)
        self.feeds[pid] = {"replay": rp, "kind": "historical", "label": "Replay of a public dataset record (simulation)", "scenario": None}
        rp.start()

    def feed_status(self, pid):
        f = self.feeds.get(pid)
        if not f: return None
        r = f["replay"]
        return {"kind": f["kind"], "label": f["label"], "scenario": f["scenario"], "status": r.status, "position": r.position, "total": r.total, "error": r.last_error}

    def pause_feed(self, pid):
        if pid in self.feeds: self.feeds[pid]["replay"].pause()

    def resume_feed(self, pid):
        if pid in self.feeds: self.feeds[pid]["replay"].resume()

    def stop_feed(self, pid):
        f = self.feeds.pop(pid, None)
        if f: f["replay"].stop()

    def any_feed_running(self):
        return any(f["replay"].status == "running" for f in self.feeds.values())
