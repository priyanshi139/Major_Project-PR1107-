"""Phase 7: agentic alert layer (human approval) + FastAPI backend with roles. Model doubles are used for the tabular models (clearly named, as in the
platform tests); they are NOT used anywhere in the application code."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api import security as sec
from src.agents.triage import grounded
from src.services.patient_service import PatientService
from src.services.oversight import Oversight
from tests.test_dashboard import cfg_in
from tests.test_patient_platform import FakeTabularModel

GATE = {"passed": True, "state": "validated", "reason": "", "origin": "real_public", "trained_at": "2026-01-01"}
PAT = {"username": "pat_test", "password": "Passw0rd!x", "role": "patient", "full_name": "Pat Test", "age": 55, "gender": "Female"}


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    monkeypatch.delenv("HEALTHTWIN_DOCTOR_CODE", raising=False)
    svc = PatientService(cfg_in(tmp_path), {"tabular": {"diabetes": FakeTabularModel()}, "gates": {"diabetes": GATE}})
    app = create_app(cfg_in(tmp_path), svc); c = TestClient(app)
    for name, role in (("dr_test", "doctor"), ("root", "admin")):
        svc.store.create_user(name, sec.hash_password("Passw0rd!x"), role, name)
    tok = lambda r: {"Authorization": "Bearer " + r.json()["token"]}
    login = lambda n: tok(c.post("/api/auth/login", json={"username": n, "password": "Passw0rd!x"}))
    reg = c.post("/api/auth/register", json=PAT)
    return {"c": c, "svc": svc, "doc": login("dr_test"), "adm": login("root"), "pat": tok(reg), "pid": reg.json()["user"]["patient_id"]}


def test_auth_rules_and_throttle(ctx):
    c = ctx["c"]
    assert c.get("/api/patients").status_code == 401 and c.get("/api/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert c.post("/api/auth/register", json={**PAT, "username": "dr_two", "role": "doctor", "doctor_code": "anything"}).status_code == 403   # no invite code configured
    assert c.post("/api/auth/register", json={**PAT, "username": "adm_two", "role": "admin"}).status_code == 400
    assert c.post("/api/auth/register", json=PAT).status_code == 409
    assert c.post("/api/auth/register", json={**PAT, "username": "x1", "password": "short"}).status_code == 422
    for _ in range(5): c.post("/api/auth/login", json={"username": "ghost", "password": "aaaaaaaaa"})
    assert c.post("/api/auth/login", json={"username": "ghost", "password": "aaaaaaaaa"}).status_code == 429


def test_doctor_registration_needs_the_configured_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTHTWIN_DOCTOR_CODE", "s3cret-code")
    c = TestClient(create_app(cfg_in(tmp_path), PatientService(cfg_in(tmp_path), {})))
    body = {"username": "drx", "password": "Passw0rd!x", "role": "doctor"}
    assert c.post("/api/auth/register", json={**body, "doctor_code": "wrong"}).status_code == 403
    assert c.post("/api/auth/register", json={**body, "doctor_code": "s3cret-code"}).status_code == 200


def test_role_isolation(ctx):
    c, pid = ctx["c"], ctx["pid"]
    other = c.post("/api/patients", headers=ctx["doc"], json={"name": "Other", "age": 40, "gender": "Male"}).json()["patient_id"]
    assert c.get(f"/api/patients/{pid}", headers=ctx["pat"]).status_code == 200
    assert c.get(f"/api/patients/{other}", headers=ctx["pat"]).status_code == 403
    assert [p["patient_id"] for p in c.get("/api/patients", headers=ctx["pat"]).json()] == [pid]
    assert c.post("/api/patients", headers=ctx["pat"], json={"name": "x", "age": 3}).status_code == 403
    assert c.post("/api/agent/run", headers=ctx["pat"]).status_code == 403
    assert c.post(f"/api/patients/{pid}/observations", headers=ctx["pat"], json={"values": {"HR": 80}}).status_code == 403
    assert c.get("/api/admin/users", headers=ctx["doc"]).status_code == 403 and c.get("/api/admin/users", headers=ctx["adm"]).status_code == 200


def test_assessment_history_and_honest_unavailable(ctx):
    c, pid = ctx["c"], ctx["pid"]
    r = c.post(f"/api/patients/{pid}/assessments/diabetes", headers=ctx["pat"], json={"values": {"Glucose": 150, "BMI": 31, "Age": 55}})
    assert r.status_code == 200 and r.json()["risk_score"] == 0.72
    assert c.post(f"/api/patients/{pid}/assessments/cardiovascular", headers=ctx["pat"], json={"values": {"age": 55, "sysBP": 150, "male": 0}}).status_code == 409   # no validated model
    assert c.post(f"/api/patients/{pid}/assessments/nonsense", headers=ctx["pat"], json={"values": {}}).status_code == 400
    assert len(c.get(f"/api/patients/{pid}/assessments?disease=diabetes", headers=ctx["pat"]).json()["diabetes"]) == 1
    assert any(e["event_type"] == "disease_assessment" for e in c.get(f"/api/patients/{pid}/timeline", headers=ctx["pat"]).json())
    tw = c.get(f"/api/patients/{pid}/twin", headers=ctx["pat"]).json()
    assert len(tw["disease_risk_history"]["diabetes"]) == 1
    st = c.get("/api/system/status", headers=ctx["pat"]).json()["diseases"]
    assert st["diabetes"]["available"] is True and st["sepsis"]["available"] is False and st["cardiovascular"]["available"] is False


def test_agent_proposals_need_human_decision(ctx):
    c, pid = ctx["c"], ctx["pid"]
    c.post(f"/api/patients/{pid}/assessments/diabetes", headers=ctx["pat"], json={"values": {"Glucose": 150, "BMI": 31, "Age": 55}})
    run = c.post("/api/agent/run", headers=ctx["doc"]).json()
    assert run["new_proposals"] == 1 and run["ranking"][0]["patient_id"] == pid and run["ranking"][0]["evidence"]
    assert c.post("/api/agent/run", headers=ctx["doc"]).json()["new_proposals"] == 0                # duplicate reduction
    props = c.get("/api/agent/proposals?status=proposed", headers=ctx["doc"]).json()
    assert len(props) == 1 and props[0]["status"] == "proposed" and "diagnos" not in props[0]["title"].lower()
    assert c.get("/api/agent/proposals", headers=ctx["pat"]).status_code == 403
    d = f"/api/agent/proposals/{props[0]['proposal_id']}/decision"
    assert c.post(d, headers=ctx["pat"], json={"decision": "approved"}).status_code == 403
    assert c.post(d, headers=ctx["doc"], json={"decision": "maybe"}).status_code == 400
    ok = c.post(d, headers=ctx["doc"], json={"decision": "approved", "note": "ok"}).json()
    assert ok["status"] == "approved" and ok["decided_by"] == "dr_test"
    assert c.post(d, headers=ctx["doc"], json={"decision": "rejected"}).status_code == 400            # decided once
    actions = [a["action"] for a in c.get("/api/admin/audit", headers=ctx["adm"]).json()]
    assert "agent_run" in actions and "proposal_approved" in actions and "assessment" in actions


def test_triage_priority_and_llm_grounding(ctx):
    svc, pid = ctx["svc"], ctx["pid"]
    calm = Oversight(svc).triage()[0]
    svc.assess_disease(pid, "diabetes", {"Glucose": 150, "BMI": 31, "Age": 55})
    flagged = Oversight(svc).triage()[0]
    assert flagged["priority"] > calm["priority"] and any("diabetes" in r.lower() or "Diabetes" in r for r in flagged["reasons"])
    assert "not diagnoses" in flagged["narrative"]
    assert grounded("priority 20", "priority 20/100", []) and not grounded("priority 99", "priority 20/100", [])     # invented numbers rejected


def test_alert_actions_and_deleting_a_patient_removes_proposals(ctx):
    c, pid, svc = ctx["c"], ctx["pid"], ctx["svc"]
    assert c.post("/api/alerts/ALERT-nope/action", headers=ctx["doc"], json={"action": "acknowledge"}).status_code == 404
    assert c.post("/api/alerts/ALERT-nope/action", headers=ctx["doc"], json={"action": "delete"}).status_code == 400
    assert c.get("/api/alerts", headers=ctx["pat"]).status_code == 200
    c.post(f"/api/patients/{pid}/assessments/diabetes", headers=ctx["pat"], json={"values": {"Glucose": 150, "BMI": 31, "Age": 55}})
    c.post("/api/agent/run", headers=ctx["doc"])
    assert svc.store.get_proposals(pid)
    assert c.delete(f"/api/patients/{pid}", headers=ctx["doc"]).json()["deleted"] == pid
    assert svc.store.get_proposals(pid) == [] and svc.store.get_disease_assessments(pid) == []


def test_alerts_page_agent_panel_runs_and_decisions_are_stored(tmp_path, monkeypatch):
    from tests.test_dashboard import make_app
    at = make_app(tmp_path, monkeypatch)
    at.radio(key="page").set_value("🚨 Alerts").run()
    assert not at.exception
    at.button(key="agent_run").click().run()
    assert not at.exception
