"""Session + dashboard startup tests. SYNTHETIC data only."""
import time, yaml
from pathlib import Path
import pytest
from src.utils.config import load_config
from src.simulation.session import ReplaySession, load_assets


def cfg_in(tmp_path):
    cfg = load_config()
    for k in ("processed_dir", "artifacts_dir", "reports_dir", "raw_dir", "sample_dir"):
        cfg["paths"][k] = str(tmp_path / k)
    cfg["paths"]["database"] = str(tmp_path / "t.db")
    return cfg


def wait_for(cond, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond(): return True
        time.sleep(0.02)
    return False


def test_load_assets_missing_is_graceful(tmp_path):
    a = load_assets(cfg_in(tmp_path))
    assert a["df"] is None and a["model"] is None and a["data_error"] and a["model_error"]


def test_session_requires_patient_and_blocks_real_without_data(tmp_path):
    s = ReplaySession(cfg_in(tmp_path), load_assets(cfg_in(tmp_path)))
    assert s.status == "no_patient"
    with pytest.raises(RuntimeError): s.start()
    with pytest.raises(RuntimeError): s.select_subject("sepsis:p000001")
    assert s.subject_choices() == []


def test_session_synthetic_step_snapshot_and_reset(tmp_path):
    cfg = cfg_in(tmp_path); s = ReplaySession(cfg, load_assets(cfg)); s.select_synthetic()
    snap = s.snapshot(); assert snap["has_data"] is False and snap["is_synthetic"]
    for _ in range(12): s.step()
    snap = s.snapshot()
    assert snap["has_data"] and snap["state"]["n_observations"] == 12 and snap["position"] == 12
    assert snap["model_name"] is None and snap["model_error"]          # no model -> still works
    s.reset(); assert s.snapshot()["has_data"] is False and s.status == "idle"


def test_session_background_replay_pause_resume_and_ack(tmp_path):
    cfg = cfg_in(tmp_path); cfg["replay"] = {"base_interval_seconds": 0.01}
    s = ReplaySession(cfg, load_assets(cfg)); s.select_synthetic()
    s.start(); assert wait_for(lambda: s.replay.position >= 3)
    s.pause(); time.sleep(0.1); p = s.replay.position; time.sleep(0.15); assert s.replay.position == p
    s.resume(); assert wait_for(lambda: s.status == "finished")
    snap = s.snapshot(); assert snap["position"] == snap["total"] == 48
    if snap["alerts"]:
        a = snap["alerts"][0]["alert_id"]; s.alerts.acknowledge(a)
        assert [x for x in s.snapshot()["alerts"] if x["alert_id"] == a][0]["status"] == "acknowledged"


def test_switching_subject_resets_previous(tmp_path):
    cfg = cfg_in(tmp_path); s = ReplaySession(cfg, load_assets(cfg))
    s.select_synthetic(1)
    for _ in range(5): s.step()
    s.select_synthetic(2); assert s.snapshot()["has_data"] is False


# ---- Streamlit AppTest: the patient-centric app (all data synthetic) ----
import json
PAGES = ["🏠 Overview", "👥 Patients", "🪪 Profile", "🧬 Twin", "📡 Live", "🔬 Prediction", "🕒 Timeline", "🔮 What-if", "💡 Insights", "🚨 Alerts", "🗄 Data"]


def make_app(tmp_path, monkeypatch, with_model=False):
    from streamlit.testing.v1 import AppTest
    cfg = cfg_in(tmp_path)
    if with_model:
        monkeypatch.setenv("HEALTHTWIN_ALLOW_TEST_MODELS", "1")      # synthetic-trained model: loadable ONLY in explicit test mode, never 'available'
        from tests.test_sepsis_phase3 import synth
        from src.preprocessing.sepsis_preprocessing import patient_wise_split
        from src.models.train_sepsis import run_training
        df = synth(); sp = patient_wise_split(df, seed=1); proc = Path(cfg["paths"]["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
        df.to_parquet(proc / "sepsis_raw_combined.parquet"); (proc / "sepsis_splits.json").write_text(json.dumps({"splits": sp}))
        run_training(df, sp, Path(cfg["paths"]["artifacts_dir"]) / "sepsis", cfg["paths"]["reports_dir"], 1)
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(cfg)); monkeypatch.setenv("HEALTHTWIN_CONFIG", str(p))
    root = Path(__file__).resolve().parents[1]
    return AppTest.from_file(str(root / "app.py"), default_timeout=120).run()


def walk(at):
    bad = []
    for page in PAGES:
        at.radio(key="page").set_value(page).run()
        if at.exception: bad.append((page, at.exception[0].value[:150]))
    return bad


def test_app_starts_with_no_data_no_model_no_patients(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    assert not at.exception and walk(at) == []                    # every page has a meaningful empty state


def test_demo_patients_are_synthetic_labelled_and_pages_render(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.radio(key="page").set_value("🏠 Overview").run(); at.button(key="demo_🏠 Overview").click().run()
    assert [o.split(" · ")[0] for o in at.sidebar.selectbox[0].options] == ["HT-0001", "HT-0002", "HT-0003"]   # sidebar refreshed
    assert all("SYNTHETIC" in o for o in at.sidebar.selectbox[0].options)
    assert walk(at) == []
    at.radio(key="page").set_value("🧬 Twin").run()
    assert any("SYNTHETIC" in str(x.value) for x in at.markdown)


def test_register_patient_and_enter_observation_through_forms(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch)
    at.radio(key="page").set_value("👥 Patients").run()
    at.text_input(key="reg_name").set_value("Ward test"); at.number_input(key="reg_age").set_value(71)
    at.multiselect(key="reg_conditions").set_value(["Hypertension", "Type 2 diabetes"]); at.button(key="btn_register").click().run()
    assert not at.exception and "Ward test" in at.sidebar.selectbox[0].options[0]
    svc = at.session_state["svc"]; p = svc.get_patient("HT-0001")
    assert p["age"] == 71 and p["conditions"] == ["Hypertension", "Type 2 diabetes"] and p["synthetic"] is True
    at.radio(key="page").set_value("📡 Live").run()
    at.number_input(key="obs_HR_0").set_value(88.0); at.number_input(key="obs_Lactate_0").set_value(1.4); at.button(key="btn_save_obs").click().run()
    assert not at.exception and any("Saved at hour 0" in s.value for s in at.success)
    vals = svc.store.get_observations("sepsis:HT-0001")[0]["values"]
    assert vals["HR"] == 88.0 and vals["Lactate"] == 1.4 and vals["Resp"] is None
    at.number_input(key="obs_HR_1").set_value(90.0)
    at.time_input(key="obs_time_1").set_value(__import__("datetime").time(0, 0)); at.button(key="btn_save_obs").click().run()
    assert any("later than the last observation" in e.value for e in at.error)             # timestamp must move forward
    assert svc.ensure_twin("HT-0001").n_observations == 1 and walk(at) == []              # incl. What-if with a single observation


def test_profile_edit_and_delete(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch); svc = at.session_state["svc"]
    at.radio(key="page").set_value("🏠 Overview").run(); at.button(key="demo_🏠 Overview").click().run()
    active = at.sidebar.selectbox[0].value                                                  # the demo button selects the 2nd patient
    at.radio(key="page").set_value("🪪 Profile").run()
    at.number_input(key="prof_age").set_value(80); at.button(key="btn_save_profile").click().run()
    assert not at.exception and svc.get_patient(active)["age"] == 80
    at.checkbox(key="del_confirm").set_value(True); at.button(key="btn_delete").click().run()
    assert svc.get_patient(active) is None and walk(at) == []


def test_whatif_alerts_and_import_with_real_model(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch, with_model=True); svc = at.session_state["svc"]
    at.radio(key="page").set_value("🗄 Data").run(); at.button(key="btn_import").click().run()
    assert not at.exception and svc.get_patient("HT-0001")["origin"] == "physionet_2019_record"
    at.radio(key="page").set_value("🗄 Data").run(); at.button(key="btn_demo").click().run()
    pid = at.sidebar.selectbox[0].value; n_obs = svc.ensure_twin(pid).n_observations
    at.radio(key="page").set_value("🔮 What-if").run()
    hr = svc.snapshot(pid)["state"]["hour"]; at.number_input(key=f"wi_MAP_{hr}").set_value(95.0); at.button(key="btn_whatif").click().run()
    assert not at.exception and svc.store.get_whatifs(pid)[0]["changes"] == {"MAP": 95.0}
    assert any("SIMULATED" in str(x.value) for x in at.markdown) and svc.ensure_twin(pid).n_observations == n_obs
    at.radio(key="page").set_value("🔬 Prediction").run(); assert not at.exception
    assert any("Why this score" in str(x.value) for x in at.markdown)                      # explanation shown for the selected model
    at.radio(key="page").set_value("🚨 Alerts").run(); at.radio(key="alert_scope").set_value("All patients").run()
    assert svc.all_alerts(), "the toy model should have produced automatic alerts for the deteriorating demo patients"
    at.radio(key="alert_action").set_value("Acknowledge"); at.button(key="btn_ack").click().run()
    assert not at.exception and any(a["status"] == "acknowledged" for a in svc.all_alerts())
    assert walk(at) == []
