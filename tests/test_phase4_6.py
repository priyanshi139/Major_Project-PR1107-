"""Tests for Digital Twin, DB, anomaly, alerts, replay, engine. All data SYNTHETIC."""
import time, math
import numpy as np, pytest
from src.data_ingestion.sepsis_loader import FEATURES, VITALS
from src.digital_twin.models import Observation
from src.digital_twin.twin import DigitalTwin
from src.digital_twin.engine import TwinEngine
from src.anomaly_detection.rolling import RollingBaselineDetector
from src.alerts.manager import AlertManager
from src.database.db import Database
from src.simulation.replay import DataReplayEngine
from src.simulation.builders import make_synthetic_observations


def ob(h, sid="sepsis:t1", **v):
    vals = {k: None for k in FEATURES}; vals.update(v)
    return Observation(sid, "sepsis", h, vals)


def twin(sid="sepsis:t1"):
    return DigitalTwin(sid, "sepsis", FEATURES, VITALS)


# ---------- twin ----------
def test_twin_updates_and_ffill_with_age():
    t = twin(); t.add_observation(ob(0, HR=80.0)); t.add_observation(ob(1)); t.add_observation(ob(2, Resp=18.0))
    s = t.state()
    assert s.latest_values["HR"] == 80.0 and s.hours_since_measured["HR"] == 2 and s.hours_since_measured["Resp"] == 0
    assert s.n_observations == 3 and s.observations_used["last_hour"] == 2


def test_twin_rejects_out_of_order_duplicate_and_foreign():
    t = twin(); t.add_observation(ob(5, HR=70.0))
    with pytest.raises(ValueError): t.add_observation(ob(4, HR=70.0))
    with pytest.raises(ValueError): t.add_observation(ob(5, HR=70.0))
    with pytest.raises(ValueError): t.add_observation(ob(6, sid="sepsis:other"))


def test_namespace_enforced():
    with pytest.raises(ValueError): Observation("ecg:100", "sepsis", 0, {})
    eng = TwinEngine("sepsis", FEATURES, VITALS)
    with pytest.raises(ValueError): eng.process(Observation("ecg:100", "ecg", 0, {}))


def test_twin_missing_data_quality_and_history_state_at_reset():
    t = twin(); t.add_observation(ob(0, HR=80.0, Resp=16.0, Temp=37.0, O2Sat=97.0, SBP=120.0))
    assert t.state().data_quality == "good"
    t.add_observation(ob(1)); assert t.state().data_quality == "poor"
    assert t.state_at(0)["hour"] == 0 and t.state_at(5)["hour"] == 1
    with pytest.raises(KeyError): t.state_at(-1)
    assert len(t.history_df(upto_hour=0)) == 1 and list(t.history_df().columns[:2]) == ["subject_id", "hour"]
    t.reset(); assert t.state().n_observations == 0 and t.status == "no_data"


# ---------- anomaly ----------
def run_hr(values):
    t, d, flagged = twin(), RollingBaselineDetector(), []
    for h, v in enumerate(values):
        o = ob(h, HR=v); t.add_observation(o); flagged.append(d.detect(t, o))
    return flagged


def test_naturally_high_baseline_not_flagged_but_jump_is():
    rng = np.random.default_rng(0)
    high = list(110 + rng.normal(0, 1.5, 20)) + [112.0]
    assert run_hr(high)[-1] == []                       # high but normal FOR THIS PATIENT
    low_then_jump = list(70 + rng.normal(0, 1.5, 20)) + [130.0]
    f = run_hr(low_then_jump)[-1]
    assert f and f[0]["variable"] == "HR" and f[0]["direction"] == "above" and "not a confirmed" in f[0]["explanation"]


def test_anomaly_needs_baseline_and_ignores_missing():
    assert all(x == [] for x in run_hr([70, 71, 200]))   # too little history
    t, d = twin(), RollingBaselineDetector()
    for h in range(10): t.add_observation(ob(h, HR=70.0 + (h % 2)))
    o = ob(10); t.add_observation(o); assert d.detect(t, o) == []   # missing value never flagged


# ---------- alerts ----------
RISK = lambda s, thr=0.5: {"risk_score": s, "threshold": thr, "n_observations_used": 5, "last_hour_used": 4,
                           "top_factors": [{"feature": "HR_roll6_mean", "contribution": 1.2}]}


def test_alert_severity_cooldown_escalation_ack():
    am = AlertManager({"critical_threshold": 0.9, "cooldown_hours": 6})
    assert am.evaluate("sepsis:t1", "sepsis", 1, RISK(0.3)) == []
    a1 = am.evaluate("sepsis:t1", "sepsis", 2, RISK(0.7))
    assert a1[0]["severity"] == "warning" and "not a diagnosis" in a1[0]["explanation"]
    assert am.evaluate("sepsis:t1", "sepsis", 4, RISK(0.7)) == [] and am.suppressed == 1   # duplicate suppressed
    assert am.evaluate("sepsis:t1", "sepsis", 5, RISK(0.95))[0]["severity"] == "critical"  # escalation passes
    assert am.evaluate("sepsis:t1", "sepsis", 9, RISK(0.7))                                  # cooldown over
    am.acknowledge(a1[0]["alert_id"], "dr_x")
    assert a1[0]["alert_id"] not in [x["alert_id"] for x in am.active()]
    assert len(am.history()) == 3
    with pytest.raises(KeyError): am.acknowledge("nope")


def test_anomaly_alert_is_info():
    am = AlertManager()
    an = [{"variable": "HR", "z": 4.0, "explanation": "HR unusual."}]
    out = am.evaluate("sepsis:t1", "sepsis", 3, None, an)
    assert out[0]["severity"] == "info" and out[0]["type"] == "anomaly"


# ---------- database ----------
def test_database_roundtrip_and_clear(tmp_path):
    db = Database(tmp_path / "x.db")
    db.save_observation(ob(0, HR=float("nan"))); db.save_observation(ob(1, HR=80.0))
    db.save_risk("sepsis:t1", "sepsis", 1, {"risk_score": 0.4})
    am = AlertManager(db=db); al = am.evaluate("sepsis:t1", "sepsis", 1, RISK(0.8))[0]
    assert db.get_observations("sepsis:t1")[0]["values"]["HR"] is None
    assert db.get_risks("sepsis:t1")[0]["risk_score"] == 0.4
    am.acknowledge(al["alert_id"])
    assert db.get_alerts(status="acknowledged")[0]["alert_id"] == al["alert_id"]
    db.clear_subject("sepsis:t1"); assert sum(db.counts().values()) == 0
    assert Database(":memory:").counts()["alerts"] == 0


# ---------- engine ----------
class FakeModel:
    def __init__(self): self.lengths = []
    def predict(self, hist):
        self.lengths.append(len(hist)); hr = hist["HR"].dropna().iloc[-1] if hist["HR"].notna().any() else 0
        s = 0.95 if hr > 100 else 0.1
        return {"subject_id": hist["subject_id"].iloc[0], "model_name": "fake", "risk_score": s, "threshold": 0.5,
                "flagged": s >= 0.5, "last_hour_used": int(hist["hour"].iloc[-1]), "n_observations_used": len(hist),
                "top_factors": []}


def test_engine_end_to_end_no_future_and_cooldown(tmp_path):
    fm, db = FakeModel(), Database(tmp_path / "e.db")
    am = AlertManager({"cooldown_hours": 6}, db)
    eng = TwinEngine("sepsis", FEATURES, VITALS, fm, RollingBaselineDetector(), am, db)
    obs = make_synthetic_observations(seed=1, n_hours=48)
    assert all("SYNTHETIC" in o.subject_id for o in obs)
    results = [eng.process(o) for o in obs]
    assert fm.lengths == list(range(1, 49))               # model only ever saw hours seen so far
    assert any(r["state"]["status"] == "high_risk_flag" for r in results)
    assert 0 < len(am.history()) and am.suppressed > 0
    assert db.counts()["observations"] == 48 and db.counts()["states"] == 48


def test_engine_without_model_and_model_failure_is_graceful():
    eng = TwinEngine("sepsis", FEATURES, VITALS, None, RollingBaselineDetector(), AlertManager())
    r = eng.process(ob(0, HR=80.0)); assert r["risk"] is None and r["model_available"] is False

    class Bad:
        def predict(self, h): raise RuntimeError("boom")
    eng = TwinEngine("sepsis", FEATURES, VITALS, Bad(), None, AlertManager())
    r = eng.process(ob(0, HR=80.0)); assert r["risk_error"] == "boom"


def test_engine_reset_subject():
    db = Database(); eng = TwinEngine("sepsis", FEATURES, VITALS, FakeModel(), None, AlertManager(db=db), db)
    eng.process(ob(0, HR=150.0)); eng.reset_subject("sepsis:t1")
    assert eng.get_twin("sepsis:t1").n_observations == 0 and sum(db.counts().values()) == 0


# ---------- replay ----------
def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond(): return True
        time.sleep(0.01)
    return False


def test_replay_order_validation():
    with pytest.raises(ValueError): DataReplayEngine([ob(2), ob(1)], lambda o: None)
    with pytest.raises(ValueError): DataReplayEngine([ob(1), ob(1)], lambda o: None)


def test_replay_runs_in_order_to_finish():
    seen = []; rp = DataReplayEngine([ob(h) for h in range(6)], lambda o: seen.append(o.hour), base_interval=0.01)
    rp.start(); assert wait_for(lambda: rp.status == "finished"); assert seen == list(range(6))
    with pytest.raises(RuntimeError): rp.start()


def test_replay_pause_resume_reset_speed():
    seen, resets = [], []
    rp = DataReplayEngine([ob(h) for h in range(40)], lambda o: seen.append(o.hour),
                          on_reset=lambda: resets.append(1), base_interval=0.03)
    rp.start(); assert wait_for(lambda: rp.position >= 2)
    rp.pause(); time.sleep(0.1); p = rp.position
    assert rp.status == "paused"; time.sleep(0.2); assert rp.position == p      # truly paused
    rp.set_speed(100.0); rp.resume(); assert wait_for(lambda: rp.status == "finished", 5)
    assert seen == list(range(40))
    rp.reset(); assert rp.position == 0 and rp.status == "idle" and resets == [1]
    with pytest.raises(ValueError): rp.set_speed(0)
    rp.start(); assert wait_for(lambda: rp.status == "finished", 5)             # can replay again after reset


def test_replay_callback_error_sets_error_status():
    def bad(o): raise RuntimeError("x")
    rp = DataReplayEngine([ob(0), ob(1)], bad, base_interval=0.0); rp.start()
    assert wait_for(lambda: rp.status == "error") and rp.last_error == "x"
