"""Tests for alert workflow, what-if, narrative, upload validation, uploaded-patient session. SYNTHETIC data only."""
import numpy as np, pandas as pd, pytest
from src.alerts.manager import AlertManager
from src.database.db import Database
from src.data_ingestion.sepsis_loader import FEATURES
from src.data_ingestion.upload import validate_patient_upload, template_csv
from src.digital_twin.whatif import apply_adjustments, compare_scenarios
from src.insights.narrative import summarize_patient, explain_prediction, explain_whatif, risk_trend
from src.simulation.session import ReplaySession, load_assets
from src.models.train_sepsis import run_training
from src.models.sepsis_inference import SepsisRiskModel
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from tests.test_sepsis_phase3 import synth
from tests.test_dashboard import cfg_in

RISK = lambda s: {"risk_score": s, "threshold": 0.5, "n_observations_used": 5, "last_hour_used": 4, "top_factors": []}


# ---------- alert workflow ----------
def test_alert_acknowledge_escalate_resolve_and_audit():
    am = AlertManager({"cooldown_hours": 6}); a = am.evaluate("sepsis:t", "sepsis", 1, RISK(0.7))[0]
    am.escalate(a["alert_id"], "senior", "dr_x", "worsening")
    assert a["status"] == "escalated" and a["escalated_to"] == "senior" and am.pending()[0]["alert_id"] == a["alert_id"]
    am.acknowledge(a["alert_id"], "dr_y"); am.resolve(a["alert_id"], "dr_y", "done")
    assert [e["status"] for e in a["history"]] == ["active", "escalated", "acknowledged", "resolved"]
    with pytest.raises(ValueError): am.acknowledge(a["alert_id"])           # resolved is final
    assert am.pending() == []


def test_critical_alert_auto_escalates_but_warning_does_not(tmp_path):
    db = Database(tmp_path / "x.db"); am = AlertManager({"escalate_after_hours": 3, "cooldown_hours": 50}, db)
    c = am.evaluate("sepsis:t", "sepsis", 1, RISK(0.95))[0]
    w = am.evaluate("sepsis:t", "sepsis", 1, RISK(0.7))[0]
    am.evaluate("sepsis:t", "sepsis", 3, None); assert c["status"] == "active"
    am.evaluate("sepsis:t", "sepsis", 4, None)
    assert c["status"] == "escalated" and "no real notification" in c["escalated_to"] and w["status"] == "active"
    assert db.get_alerts(status="escalated")[0]["alert_id"] == c["alert_id"]


# ---------- what-if ----------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wi"); df = synth(); sp = patient_wise_split(df, seed=1)
    run_training(df, sp, tmp / "a", tmp / "r", seed=1)
    return df, sp, SepsisRiskModel(tmp / "a")


def test_apply_adjustments_only_touches_measured_values_from_start():
    h = pd.DataFrame({"subject_id": "s", "hour": range(6), "HR": [80, np.nan, 82, 84, 86, 88.0]})
    out = apply_adjustments(h, {"HR": -10.0, "Resp": 0.0}, 3)
    assert out["HR"][0] == 80 and np.isnan(out["HR"][1]) and out["HR"][2] == 82      # missing stays missing
    assert out["HR"].tolist()[3:] == [74, 76, 78] and h["HR"][3] == 84       # original untouched
    assert apply_adjustments(h, {"HR": -500.0}, 0)["HR"].dropna().min() == 20  # clipped to physiological limit
    with pytest.raises(ValueError): apply_adjustments(h, {"Age": 5.0}, 0)


def test_whatif_no_change_identical_and_shift_changes_score(trained):
    df, sp, model = trained; pid = sp["test"][0]
    h = df[df.subject_id == pid].iloc[:20].reset_index(drop=True)
    same = compare_scenarios(model, h, {"HR": 0.0}, 10)
    assert same["max_abs_delta"] == pytest.approx(0.0, abs=1e-12)
    shifted = compare_scenarios(model, h, {"HR": 5.0}, 10)
    assert shifted["max_abs_delta"] > 0 and shifted["baseline"]["hour"].iloc[-1] == 19
    assert "not a prediction" in shifted["note"].lower()
    with pytest.raises(RuntimeError): compare_scenarios(None, h, {}, 0)
    with pytest.raises(ValueError): compare_scenarios(model, h.iloc[0:0], {}, 0)
    # scenario starts at hour 10; replayed window starts at hour 8 -> hours 8 and 9 must be unchanged (no leakage backwards)
    assert shifted["baseline"]["hour"].tolist()[:2] == [8, 9]
    assert shifted["baseline"]["risk_score"].values[:2] == pytest.approx(shifted["scenario"]["risk_score"].values[:2])


# ---------- upload ----------
def good_df():
    return pd.DataFrame({"hour": range(8), "HR": [80, 81, 82, 83, 84, 85, 86, 87.0], "Resp": [16] * 8, "Foo": 1})


def test_upload_valid_with_warnings_and_template():
    clean, err, warn = validate_patient_upload(good_df(), "my file.csv")
    assert err == [] and clean["subject_id"].iloc[0] == "sepsis:UPLOADED_my_file_csv"
    assert set(FEATURES) <= set(clean.columns) and any("unknown columns" in w for w in warn)
    assert template_csv().startswith("hour,HR")


def test_upload_rejects_bad_input():
    assert validate_patient_upload(pd.DataFrame(), "x")[0] is None
    assert validate_patient_upload(pd.DataFrame({"foo": [1]}), "x")[0] is None
    d = good_df(); d.loc[3, "hour"] = 1
    assert "increasing" in validate_patient_upload(d, "x")[1][0]
    d = good_df(); d["hour"] = d["hour"] + 0.5
    assert "whole numbers" in validate_patient_upload(d, "x")[1][0]
    big = pd.DataFrame({"hour": range(600), "HR": 80.0}); assert "Too many rows" in validate_patient_upload(big, "x")[1][0]


def test_upload_out_of_range_and_text_become_missing():
    d = good_df(); d.loc[2, "HR"] = 900; d["Resp"] = d["Resp"].astype(object); d.loc[1, "Resp"] = "abc"
    clean, err, warn = validate_patient_upload(d, "x")
    assert err == [] and np.isnan(clean.loc[2, "HR"]) and np.isnan(clean.loc[1, "Resp"]) and len(warn) >= 3
    only_bad = pd.DataFrame({"hour": [0, 1], "HR": [900, 901]})
    assert validate_patient_upload(only_bad, "x")[0] is None


# ---------- session with uploaded patient + model + narrative ----------
def test_session_uploaded_patient_what_if_and_insights(trained, tmp_path):
    df, sp, model = trained; cfg = cfg_in(tmp_path)
    s = ReplaySession(cfg, {**load_assets(cfg), "model": model})
    clean, _, _ = validate_patient_upload(pd.DataFrame({"hour": range(15), "HR": np.linspace(80, 95, 15), "Resp": 16.0,
                                                        "SepsisLabel": [0] * 15}), "demo")
    s.select_uploaded(clean)
    assert s.has_uploaded and s.provenance == "uploaded"
    for _ in range(15): s.step()
    snap = s.snapshot()
    assert snap["provenance"] == "uploaded" and snap["state"]["n_observations"] == 15 and snap["state"]["risk"]
    txt = " ".join(summarize_patient(snap)); assert "user" in txt.lower() and "not medical advice" in txt
    assert "threshold" in " ".join(explain_prediction(snap))
    cmp = s.whatif({"HR": -10.0}, 10); assert " ".join(explain_whatif(cmp)).count("Model sensitivity") == 1
    s.select_synthetic(); assert s.provenance == "synthetic" and s.snapshot()["has_data"] is False
    s.reselect_uploaded(); assert s.provenance == "uploaded"
    with pytest.raises(RuntimeError): ReplaySession(cfg, load_assets(cfg)).whatif({}, 0)


def test_narrative_without_data_and_trend():
    assert "No observations" in summarize_patient({"has_data": False})[0]
    assert explain_prediction({"has_data": False})[0].startswith("No prediction")
    assert risk_trend([{"risk_score": x} for x in [0.1, 0.2, 0.3, 0.4]])[0] == "rising"
    assert risk_trend([{"risk_score": 0.1}])[0].startswith("not enough")


# ---------- live entry + patient cards (session level) ----------
def test_live_entry_session(trained, tmp_path):
    df, sp, model = trained; cfg = cfg_in(tmp_path)
    s = ReplaySession(cfg, {**load_assets(cfg), "model": model})
    with pytest.raises(RuntimeError): s.add_reading({"HR": 90.0})           # need to start a session first
    s.start_live("Ward 3 / bed 2")
    assert s.provenance == "live_entry" and s.subject_id == "sepsis:LIVE_Ward_3_bed_2" and s.status == "live entry"
    with pytest.raises(ValueError): s.add_reading({})                        # nothing entered
    with pytest.raises(ValueError): s.add_reading({"HR": 900.0})             # out of range
    r0 = s.add_reading({"HR": 90.0, "Resp": 18.0}); r1 = s.add_reading({"HR": 92.0})
    assert (r0["state"]["hour"], r1["state"]["hour"]) == (0, 1) and r1["risk"] is not None
    snap = s.snapshot(); assert snap["position"] == snap["total"] == 2 and snap["provenance"] == "live_entry"
    with pytest.raises(RuntimeError): s.start()                              # replay controls do not apply
    s.start_live("other"); assert s.snapshot()["has_data"] is False


def test_patient_cards_are_deterministic_and_mixed(trained, tmp_path):
    df, sp, model = trained; cfg = cfg_in(tmp_path)
    s = ReplaySession(cfg, {**load_assets(cfg), "df": df, "splits": sp, "model": model})
    a, b = s.patient_cards(6), s.patient_cards(6)
    assert [x["id"] for x in a] == [x["id"] for x in b] and len({x["id"] for x in a}) == len(a)
    assert all(x["id"] in sp["test"] for x in a) and any(x["sepsis_recorded"] for x in a) and any(not x["sepsis_recorded"] for x in a)
    assert ReplaySession(cfg, load_assets(cfg)).patient_cards() == []
