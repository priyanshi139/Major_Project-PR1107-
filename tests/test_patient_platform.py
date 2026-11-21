"""Patient-centric platform tests. All data are SYNTHETIC (generated in code); toy model trained on synthetic data."""
import time
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytest
from src.data_ingestion.schema import validate_values, SPECS
from src.data_ingestion.sepsis_loader import FEATURES, VITALS, LABS
from src.models.train_sepsis import run_training
from src.models.sepsis_inference import SepsisRiskModel
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from src.services.patient_service import PatientService, validate_profile
from src.simulation.feed import generate_feed
from tests.test_sepsis_phase3 import synth
from tests.test_dashboard import cfg_in

T0 = datetime(2026, 1, 1, 8, 0, 0)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy"); df = synth(); sp = patient_wise_split(df, seed=1)
    run_training(df, sp, tmp / "a", tmp / "r", seed=1)
    return df, sp, SepsisRiskModel(tmp / "a")


def make(tmp_path, assets=None):
    return PatientService(cfg_in(tmp_path), assets or {})


def reg(svc, **kw):
    p = {"name": "Test patient", "age": 62, "gender": "Male", "conditions": ["Hypertension"], "monitoring_start": T0.isoformat()}; p.update(kw)
    return svc.register_patient(p)


class FakeModel:
    """Deterministic stand-in so alert rules can be tested precisely. predict() sees only the history it is given."""
    name, threshold = "fake", 0.5
    class pre:  # noqa
        medians_ = {v: 0.0 for v in VITALS + LABS}
    def predict(self, hist):
        hr = hist["HR"].dropna().iloc[-1] if hist["HR"].notna().any() else 0
        s = float(np.clip((hr - 80) / 60, 0, 1))
        return {"subject_id": hist["subject_id"].iloc[0], "model_name": "fake", "risk_score": s, "threshold": 0.5, "flagged": s >= 0.5,
                "last_hour_used": int(hist["hour"].iloc[-1]), "n_observations_used": len(hist), "top_factors": []}


# ---------- schema ----------
def test_schema_covers_all_measurable_model_inputs_and_validates():
    assert list(SPECS) == VITALS + LABS
    clean, errs = validate_values({"HR": "90", "MAP": "", "pH": None, "Lactate": 2.1})
    assert clean == {"HR": 90.0, "Lactate": 2.1} and errs == []                      # blanks stay missing
    _, errs = validate_values({"HR": 900, "pH": "abc", "Foo": 1, "O2Sat": float("nan")})
    assert len(errs) == 4


# ---------- registration / profile ----------
def test_register_unique_ids_validation_and_profile_edit(tmp_path):
    svc = make(tmp_path); a, b = reg(svc), reg(svc, name="Second")
    assert (a, b) == ("HT-0001", "HT-0002") and svc.get_patient(a)["synthetic"] is True
    with pytest.raises(ValueError): validate_profile({"name": "x", "age": 200})
    with pytest.raises(ValueError): validate_profile({"age": 40, "conditions": ["Flu"]})
    assert [p["patient_id"] for p in svc.list_patients("second")] == ["HT-0002"]
    assert [p["patient_id"] for p in svc.list_patients("hyperten")] == ["HT-0001", "HT-0002"]
    svc.update_patient(a, {"age": 63, "conditions": ["Hypertension", "Type 2 diabetes"], "history": "ex-smoker"})
    p = svc.get_patient(a); assert p["age"] == 63 and "Type 2 diabetes" in p["conditions"]
    types = [e["event_type"] for e in svc.store.get_events(a)]; assert types == ["registration", "profile"]
    svc.add_observation(a, {"HR": 80}, T0)
    with pytest.raises(ValueError): svc.update_patient(a, {"monitoring_start": (T0 + timedelta(days=1)).isoformat()})


# ---------- observation entry, twin update, prediction, persistence ----------
def test_manual_observation_updates_twin_predicts_and_persists(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": model}); pid = reg(svc)
    r0 = svc.add_observation(pid, {"HR": 88, "Resp": 18, "MAP": 80, "Lactate": 1.5}, T0)
    r1 = svc.add_observation(pid, {"HR": 92}, T0 + timedelta(hours=1))
    assert (r0["hour"], r1["hour"]) == (0, 1) and r1["risk"]["risk_category"] in ("Low", "Elevated", "High", "Critical")
    sn = svc.snapshot(pid)
    assert sn["state"]["n_observations"] == 2 and sn["state"]["latest_values"]["Lactate"] == 1.5 and len(sn["risk_history"]) == 2
    rec = svc.store.get_observations(f"sepsis:{pid}")[0]["values"]
    assert rec["DBP"] is None and rec["EtCO2"] is None                               # not measured -> stored as missing, not filled
    assert rec["Age"] == 62.0 and rec["Gender"] == 1.0 and rec["ICULOS"] == 1.0       # derived from the profile, not invented
    assert "SepsisLabel" not in rec
    # restart: a brand-new service on the same database rebuilds the twin
    svc2 = PatientService(cfg_in(tmp_path), {"model": model}); sn2 = svc2.snapshot(pid)
    assert sn2["state"]["n_observations"] == 2 and sn2["state"]["latest_values"]["HR"] == 92.0
    assert [x["risk_score"] for x in sn2["risk_history"]] == pytest.approx([x["risk_score"] for x in sn["risk_history"]])
    assert sn2["obs_records"][1]["timestamp"] == (T0 + timedelta(hours=1)).isoformat() and sn2["obs_records"][0]["source"] == "manual_entry"


def test_observation_validation_and_timestamp_rules(tmp_path):
    svc = make(tmp_path); pid = reg(svc)
    for bad, msg in [({}, "at least one vital"), ({"Lactate": 2.0}, "at least one vital"), ({"HR": 900}, "plausible"), ({"HR": "x"}, "not a number")]:
        with pytest.raises(ValueError, match=msg): svc.add_observation(pid, bad, T0)
    svc.add_observation(pid, {"HR": 80}, T0 + timedelta(hours=2))
    with pytest.raises(ValueError, match="later than the last"): svc.add_observation(pid, {"HR": 80}, T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="before the monitoring start"): svc.add_observation(reg(svc), {"HR": 80}, T0 - timedelta(hours=3))
    r = svc.add_observation(pid, {"HR": 81}, T0 + timedelta(hours=2, minutes=30))     # same hour slot -> next slot, explained
    assert r["hour"] == 3 and "hour slot 3" in r["slot_note"]
    with pytest.raises(KeyError): svc.add_observation("HT-9999", {"HR": 80})


def test_no_future_data_in_predictions(tmp_path, toy):
    df, sp, model = toy; seen = []
    class Spy(FakeModel):
        def predict(self, hist): seen.append((len(hist), int(hist["hour"].max()))); return super().predict(hist)
    svc = make(tmp_path, {"model": Spy()}); pid = reg(svc)
    for i in range(5): svc.add_observation(pid, {"HR": 80 + i}, T0 + timedelta(hours=i))
    assert seen == [(i + 1, i) for i in range(5)]                                      # model only ever saw hours already received


# ---------- automatic alerts ----------
def test_alerts_generated_automatically_with_context_and_timeline(tmp_path):
    svc = make(tmp_path, {"model": FakeModel()}); pid = reg(svc)
    for i, hr in enumerate([80, 82, 81, 83]): svc.add_observation(pid, {"HR": hr}, T0 + timedelta(hours=i))
    assert svc.snapshot(pid)["alerts"] == []
    r = svc.add_observation(pid, {"HR": 125, "Resp": 20}, T0 + timedelta(hours=4))
    al = [a for a in r["new_alerts"] if a["type"] == "risk"]; assert al
    a = al[0]
    assert a["severity"] in ("warning", "critical") and a["patient_id"] == pid and a["status"] == "active" and a["domain"] == "sepsis model"
    assert a["relevant_observations"]["HR"] == 125 and a["data_timestamp"] == (T0 + timedelta(hours=4)).isoformat() and "threshold" in a["trigger"]
    assert any(e["event_type"] == "alert" for e in svc.store.get_events(pid))
    assert svc.store.get_alerts(f"sepsis:{pid}")[0]["alert_id"] == a["alert_id"]       # persisted


def test_other_alert_rules_trend_and_multi_vital_and_not_everything_is_sepsis(tmp_path):
    svc = make(tmp_path, {"model": FakeModel()}); pid = reg(svc)
    base = {"HR": 80, "Resp": 16, "MAP": 85}
    for i in range(8): svc.add_observation(pid, {k: v + (i % 2) for k, v in base.items()}, T0 + timedelta(hours=i))
    r = svc.add_observation(pid, {"HR": 82, "Resp": 40, "MAP": 50}, T0 + timedelta(hours=8))          # two+ vitals off, HR normal
    mv = [a for a in r["new_alerts"] if a["type"] == "multi_vital"]
    assert mv and mv[0]["domain"] == "vital signs" and mv[0]["severity"] == "warning" and not [a for a in r["new_alerts"] if a["type"] == "risk"]
    svc2 = make(tmp_path / "b", {"model": FakeModel()}); p2 = reg(svc2)
    for i, hr in enumerate([80, 80, 80, 80, 105]): r = svc2.add_observation(p2, {"HR": hr}, T0 + timedelta(hours=i))
    assert [a["type"] for a in r["new_alerts"] if a["type"] == "risk_rise"] == ["risk_rise"]           # rising but still below threshold


def test_alert_workflow_records_timeline_and_priority(tmp_path):
    svc = make(tmp_path, {"model": FakeModel()}); pid = reg(svc)
    svc.add_observation(pid, {"HR": 80}, T0); r = svc.add_observation(pid, {"HR": 140}, T0 + timedelta(hours=1))
    a = r["new_alerts"][0]
    svc.alert_action(a["alert_id"], "acknowledge", "seen by Dr X"); svc.alert_action(a["alert_id"], "resolve", "stable")
    assert svc.snapshot(pid)["pending_alerts"] == 0
    titles = [e["title"] for e in svc.store.get_events(pid, ["alert_action"])]; assert len(titles) == 2 and "resolved" in titles[1]
    assert svc.all_alerts()[0]["status"] == "resolved"


# ---------- what-if ----------
def test_state_whatif_uses_real_model_and_never_touches_observations(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": model}); pid = reg(svc)
    for i in range(4): svc.add_observation(pid, {"HR": 85, "MAP": 65, "SBP": 100, "Resp": 18}, T0 + timedelta(hours=i))
    before = svc.store.get_observations(f"sepsis:{pid}"); n_events = len(svc.store.get_events(pid))
    res = svc.whatif(pid, {"MAP": 75.0})
    assert res["current"]["risk_score"] == pytest.approx(svc.snapshot(pid)["risk_history"][-1]["risk_score"])
    assert res["scenario"] is not None and res["delta_score"] == pytest.approx(res["scenario"]["risk_score"] - res["current"]["risk_score"])
    row = res["table"].set_index("variable").loc["MAP"]; assert (row.current, row.simulated, row.change) == (65.0, 75.0, 10.0)
    assert svc.store.get_observations(f"sepsis:{pid}") == before and svc.ensure_twin(pid).n_observations == 4   # simulated != observed
    assert svc.store.get_whatifs(pid)[0]["simulated"] is True and svc.store.get_events(pid)[-1]["event_type"] == "whatif"
    assert len(svc.store.get_events(pid)) == n_events + 1
    with pytest.raises(ValueError): svc.whatif(pid, {"MAP": 500.0})
    assert svc.whatif(pid, {})["scenario"] is None
    no_model = make(tmp_path / "nm"); p2 = reg(no_model); no_model.add_observation(p2, {"HR": 80}, T0)
    r2 = no_model.whatif(p2, {"HR": 100.0}); assert r2["scenario"] is None and "No trained model" in r2["reason"]


# ---------- explanations / insights / forecast ----------
def test_occlusion_explanation_and_evidence_based_insights(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": model}); pid = reg(svc, baseline={"HR": 60})
    rng = np.random.default_rng(0)
    for i in range(10): svc.add_observation(pid, {"HR": 80 + rng.normal(0, 1), "O2Sat": 97 - 1.0 * i, "Resp": 16 + rng.normal(0, .3)}, T0 + timedelta(hours=i))
    svc.add_observation(pid, {"HR": 118, "O2Sat": 80, "Resp": 16}, T0 + timedelta(hours=10))
    x = svc.explain(pid); assert x["items"] and {"feature", "effect", "neutral_value"} <= set(x["items"][0]) and "SHAP is not used" in x["note"]
    ins = svc.insights(pid)
    assert ins and all(i["source"] and i["time"] for i in ins)
    texts = " ".join(i["text"] for i in ins)
    assert "Heart rate" in texts and "baseline" in texts and "usual value in the profile" in texts and "Oxygen saturation" in texts
    from src.insights.evidence import generate_insights                            # occlusion insight: unit-tested with known numbers
    sn = svc.snapshot(pid); occ = {"items": [{"feature": "HR", "effect": 0.31, "current_value": 118.0, "neutral_value": 0.0}], "hour": 10, "model_name": "m"}
    ex = [i for i in generate_insights(sn, sn["patient"], occ, "High") if i["category"] == "explanation"][0]
    assert "Heart rate = 118.0 (+0.31" in ex["text"] and "not SHAP" in ex["source"] and ex["time"] == (T0 + timedelta(hours=10)).isoformat()
    assert any(i["category"] == "scope" and "Hypertension" in i["text"] for i in ins)      # listed condition without a model is stated plainly


# ---------- timeline ----------
def test_timeline_events_and_filters(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": FakeModel()}); pid = reg(svc)
    svc.add_observation(pid, {"HR": 80}, T0); svc.add_observation(pid, {"HR": 140}, T0 + timedelta(hours=1))
    ev = svc.store.get_events(pid); types = {e["event_type"] for e in ev}
    assert {"registration", "observation", "prediction", "alert"} <= types
    assert [e["event_id"] for e in ev] == sorted(e["event_id"] for e in ev)               # chronological
    assert {e["event_type"] for e in svc.store.get_events(pid, ["alert"])} == {"alert"}
    assert all(e["source"] == "manual_entry" for e in svc.store.get_events(pid, ["observation"]))
    assert len(svc.store.get_events(pid, limit=3)) == 3 and svc.store.get_events("HT-4242") == []


# ---------- synthetic / simulated labelling ----------
def test_demo_patients_are_labelled_synthetic_everywhere(tmp_path):
    svc = make(tmp_path); pids = svc.create_demo_patients()
    assert len(pids) == 3
    for pid in pids:
        p = svc.get_patient(pid); sn = svc.snapshot(pid)
        assert p["synthetic"] is True and p["origin"] == "demo_generator" and sn["state"]["n_observations"] == 24
        assert {r["source"] for r in sn["obs_records"]} == {"synthetic_demo"}
        assert all(e["detail"].get("synthetic") for e in svc.store.get_events(pid, ["observation"]))
    assert all(s["sources"] == ["synthetic_demo"] for s in svc.summaries())


def test_simulated_feed_adds_labelled_observations_and_updates_twin(tmp_path):
    svc = make(tmp_path, {"model": FakeModel()}); pid = reg(svc)
    svc.add_observation(pid, {"HR": 80, "Resp": 16}, T0)
    svc.start_simulated_feed(pid, "deterioration", n_hours=6, interval=0.0)
    t0 = time.time()
    while svc.feed_status(pid)["status"] != "finished" and time.time() - t0 < 10: time.sleep(0.02)
    st = svc.feed_status(pid); assert st["status"] == "finished" and st["label"] == "Simulated Live Data" and st["error"] is None
    sn = svc.snapshot(pid)
    assert sn["state"]["n_observations"] == 7 and [r["source"] for r in sn["obs_records"]][1:] == ["simulated_live"] * 6
    assert sn["state"]["latest_values"]["HR"] > 90                                    # drift reached the twin
    svc.start_simulated_feed(pid, "stable", n_hours=50, interval=0.05); svc.pause_feed(pid); time.sleep(0.15)
    p = svc.feed_status(pid)["position"]; time.sleep(0.2); assert svc.feed_status(pid)["position"] == p
    svc.stop_feed(pid); assert svc.feed_status(pid) is None
    with pytest.raises(ValueError): generate_feed("bogus")


def test_feed_never_mixes_into_real_observation_labels(tmp_path):
    svc = make(tmp_path); pid = reg(svc); svc.add_observation(pid, {"HR": 80}, T0)
    svc.add_observation(pid, {"HR": 81}, T0 + timedelta(hours=1), source="simulated_live")
    assert [r["source"] for r in svc.snapshot(pid)["obs_records"]] == ["manual_entry", "simulated_live"]


# ---------- disease registry ----------
def test_disease_registry_never_reuses_sepsis_model(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": model}); reg_ = svc.registry
    assert reg_["sepsis"].status == "pipeline_test" and reg_["sepsis"].model_info["name"] == model.name      # no gate info -> never 'available'
    gate_ok = {"passed": True, "state": "validated", "reason": "", "origin": "real_public", "trained_at": "t"}
    assert make(tmp_path / "g", {"model": model, "gates": {"sepsis": gate_ok}}).registry["sepsis"].status == "available"
    for k in ("diabetes", "cardiovascular", "hypertension"):       # a sepsis model in the session must never make another disease available
        assert reg_[k].status == "unavailable" and "Not trained yet" in reg_[k].reason and reg_[k].model_info["name"] is None
    assert reg_["ecg"].patient_linked is False and reg_["ecg"].status == "unavailable"
    assert make(tmp_path / "x").registry["sepsis"].status == "unavailable"


class FakeTabularModel:
    name, threshold = "tabular-test-double", 0.4
    def predict(self, values):
        supplied = sum(v not in (None, "") for v in values.values())
        if supplied < 3:
            raise ValueError("At least 3 clinical inputs are required")
        return {"risk_score": 0.72, "raw_score": 0.72, "threshold": 0.4, "flagged": True,
                "category": "High", "n_inputs_given": supplied, "n_inputs_total": 8,
                "missing_inputs": [], "imputation_note": None, "calibration": "test calibration",
                "disclaimer": "test only"}


def test_validated_tabular_assessment_is_saved_in_patient_twin_and_restored(tmp_path):
    gate = {"passed": True, "state": "validated", "reason": "", "origin": "real_public", "trained_at": "2026-01-01"}
    assets = {"tabular": {"diabetes": FakeTabularModel()}, "gates": {"diabetes": gate}}
    svc = make(tmp_path, assets); pid = reg(svc)
    result = svc.assess_disease(pid, "diabetes", {"Glucose": 150, "BMI": 31, "Age": 62})
    assert result["risk_score"] == 0.72 and result["assessment_type"] == "cross_sectional"
    history = svc.disease_risk_history(pid, "diabetes")["diabetes"]
    assert len(history) == 1 and history[0]["risk_score"] == 0.72 and history[0]["hour"] == -1
    assert any(e["event_type"] == "disease_assessment" for e in svc.store.get_events(pid))
    svc2 = PatientService(cfg_in(tmp_path), assets)
    restored = svc2.snapshot(pid)["disease_risk_history"]["diabetes"]
    assert len(restored) == 1 and restored[0]["risk_score"] == 0.72 and restored[0]["hour"] == -1
    with pytest.raises(ValueError): svc.assess_disease(pid, "unsupported_condition", {})


# ---------- data management ----------
def test_import_historical_replay_and_upload_and_delete(tmp_path, toy):
    df, sp, model = toy; svc = make(tmp_path, {"model": FakeModel(), "df": df, "splits": sp}); sid = sp["test"][0]
    pid = svc.import_historical(sid, n_initial=10)
    p = svc.get_patient(pid); sn = svc.snapshot(pid)
    assert p["synthetic"] is False and p["origin"] == "physionet_2019_record" and p["time_basis"] == "relative"
    assert sn["state"]["n_observations"] == 10 and {r["source"] for r in sn["obs_records"]} == {"historical_dataset"} and sn["obs_records"][0]["timestamp"] is None
    assert "SepsisLabel" not in svc.store.get_observations(f"sepsis:{pid}")[0]["values"]                    # label never an input
    svc.start_historical_replay(pid, interval=0.0); t0 = time.time()
    while svc.feed_status(pid)["status"] != "finished" and time.time() - t0 < 10: time.sleep(0.02)
    assert svc.snapshot(pid)["state"]["n_observations"] == len(df[df.subject_id == sid])
    assert "historical_replay" in {r["source"] for r in svc.snapshot(pid)["obs_records"]}
    p2 = reg(svc); svc.add_observation(p2, {"HR": 80}, T0)
    up = pd.DataFrame({"hour": [1, 2, 3], "HR": [81.0, 82.0, 83.0], "Resp": 16.0}); up["subject_id"] = "x"
    assert svc.append_uploaded(p2, up) == 3 and svc.snapshot(p2)["state"]["n_observations"] == 4
    with pytest.raises(ValueError): svc.append_uploaded(p2, up)                                              # hours must move forward
    svc.delete_patient(p2); assert svc.get_patient(p2) is None and svc.store.get_events(p2) == [] and svc.store.get_observations(f"sepsis:{p2}") == []
    with pytest.raises(ValueError): svc.start_historical_replay(p2)
