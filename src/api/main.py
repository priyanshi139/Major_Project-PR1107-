"""HealthTwin FastAPI backend over the same PatientService the Streamlit app uses.   Run:  uvicorn src.api.asgi:app --port 8000
Roles: patient (own record only), doctor (all patients, alert actions, agent proposal decisions), admin (+ users, audit).
Doctor/admin accounts are NOT self-service: create them with `python scripts/create_user.py`; doctor self-registration works only if
HEALTHTWIN_DOCTOR_CODE is set. Academic prototype - not a medical device."""
import math, os
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config, resolve_path
from src.simulation.session import load_assets
from src.services.patient_service import PatientService
from src.services.oversight import Oversight
from src.api import security as sec


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame): return jsonable(o.to_dict("records"))
    if isinstance(o, pd.Series): return jsonable(o.to_dict())
    if isinstance(o, (np.floating, float)):
        f = float(o); return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, datetime): return o.isoformat()
    return o


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=60)
    role: str = "patient"
    doctor_code: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: str = "Unknown"


class LoginIn(BaseModel):
    username: str
    password: str


class PatientIn(BaseModel):
    name: str
    age: int
    gender: str = "Unknown"
    conditions: list[str] = []
    history: str = ""
    synthetic: bool = True


class ObsIn(BaseModel):
    values: dict[str, Optional[float]]
    timestamp: Optional[str] = None
    source: str = "manual_entry"


class AssessIn(BaseModel):
    values: dict[str, Optional[float]]


class ActionIn(BaseModel):
    action: str
    note: str = ""


class DecisionIn(BaseModel):
    decision: str
    note: str = ""


def create_app(cfg=None, service: PatientService = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("HEALTHTWIN_CONFIG"))
    svc = service or PatientService(cfg, load_assets(cfg))
    over = Oversight(svc)
    secret = sec.load_secret(resolve_path(cfg, "artifacts_dir"))
    throttle = sec.Throttle()
    doctor_code = os.environ.get("HEALTHTWIN_DOCTOR_CODE")
    app = FastAPI(title="HealthTwin API", version="1.0", description="Academic prototype. Not a medical device. Alerts and proposals require human review.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], allow_methods=["*"], allow_headers=["*"])
    app.state.svc = svc

    def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Not signed in")
        try:
            claims = sec.read_token(secret, authorization.split(" ", 1)[1])
        except Exception:
            raise HTTPException(401, "Session expired or invalid; please sign in again")
        u = svc.store.get_user(int(claims["sub"]))
        if not u: raise HTTPException(401, "Unknown user")
        return u

    def staff(u=Depends(current_user)):
        if u["role"] not in ("doctor", "admin"): raise HTTPException(403, "Clinician access required")
        return u

    def admin(u=Depends(current_user)):
        if u["role"] != "admin": raise HTTPException(403, "Administrator access required")
        return u

    def access(u, pid):
        if svc.store.get_patient(pid) is None: raise HTTPException(404, f"Unknown patient {pid}")
        if u["role"] == "patient" and u["patient_id"] != pid: raise HTTPException(403, "You can only access your own record")

    def pub(u): return {k: u[k] for k in ("user_id", "username", "role", "full_name", "patient_id")}

    def guard(fn):
        try:
            return jsonable(fn())
        except HTTPException: raise
        except KeyError as e: raise HTTPException(404, str(e).strip("'\""))
        except ValueError as e: raise HTTPException(400, str(e))
        except RuntimeError as e: raise HTTPException(409, str(e))

    # ---------------------------------------------------------------- auth
    @app.get("/api/health")
    def health(): return {"status": "ok", "app": "HealthTwin"}

    @app.post("/api/auth/register")
    def register(b: RegisterIn):
        if b.role not in ("patient", "doctor"): raise HTTPException(400, "Self-registration is for 'patient' (or 'doctor' with an invitation code)")
        if b.role == "doctor" and (not doctor_code or b.doctor_code != doctor_code): raise HTTPException(403, "Doctor self-registration is disabled or the invitation code is wrong")
        if svc.store.get_user_by_name(b.username): raise HTTPException(409, "Username already taken")
        pid = None
        if b.role == "patient":
            if b.age is None: raise HTTPException(400, "Age is required for patient accounts")
            pid = guard(lambda: svc.register_patient({"name": b.full_name or b.username, "age": b.age, "gender": b.gender, "conditions": [], "synthetic": False, "origin": "self_registered"}))
        uid = svc.store.create_user(b.username, sec.hash_password(b.password), b.role, b.full_name, pid)
        svc.store.audit(b.username, b.role, "register", pid)
        u = svc.store.get_user(uid)
        return {"token": sec.make_token(secret, u), "user": pub(u)}

    @app.post("/api/auth/login")
    def login(b: LoginIn):
        if throttle.blocked(b.username): raise HTTPException(429, "Too many failed sign-ins; try again in a few minutes")
        u = svc.store.get_user_by_name(b.username)
        if not u or not sec.verify_password(b.password, u["password_hash"]):
            throttle.fail(b.username); svc.store.audit(b.username, None, "login_failed"); raise HTTPException(401, "Wrong username or password")
        throttle.ok(b.username); svc.store.touch_login(u["user_id"]); svc.store.audit(u["username"], u["role"], "login")
        return {"token": sec.make_token(secret, u), "user": pub(u)}

    @app.get("/api/auth/me")
    def me(u=Depends(current_user)): return pub(u)

    # ---------------------------------------------------------------- system
    @app.get("/api/system/status")
    def system(u=Depends(current_user)):
        return jsonable({"diseases": {k: {"name": m.name, "status": m.status, "reason": m.reason, "dataset": m.dataset, "available": m.status == "available",
                                          "separate_model": True} for k, m in svc.registry.items()},
                         "gates": {k: {kk: v.get(kk) for kk in ("state", "passed", "origin", "trained_at", "reason")} for k, v in (svc.assets.get("gates") or {}).items() if v},
                         "notice": "Only models with real-data provenance and a matching evaluation report are 'available'. Not a medical device."})

    # ---------------------------------------------------------------- patients
    @app.get("/api/patients")
    def patients(u=Depends(current_user)):
        ps = svc.list_patients()
        if u["role"] == "patient": ps = [p for p in ps if p["patient_id"] == u["patient_id"]]
        return jsonable(ps)

    @app.post("/api/patients")
    def create_patient(b: PatientIn, u=Depends(staff)):
        pid = guard(lambda: svc.register_patient(b.model_dump())); svc.store.audit(u["username"], u["role"], "patient_created", pid); return {"patient_id": pid}

    @app.get("/api/patients/{pid}")
    def patient(pid: str, u=Depends(current_user)):
        access(u, pid); return jsonable(svc.get_patient(pid))

    @app.delete("/api/patients/{pid}")
    def delete_patient(pid: str, u=Depends(staff)):
        access(u, pid); svc.delete_patient(pid); svc.store.audit(u["username"], u["role"], "patient_deleted", pid); return {"deleted": pid}

    @app.get("/api/patients/{pid}/twin")
    def twin(pid: str, u=Depends(current_user)):
        access(u, pid)
        def f():
            sn = svc.snapshot(pid)
            return {k: sn[k] for k in ("patient", "has_data", "state", "risk_history", "disease_risk_history", "health_trajectory", "baseline_quality", "category", "pending_alerts")}
        return guard(f)

    @app.get("/api/patients/{pid}/timeline")
    def timeline(pid: str, u=Depends(current_user)):
        access(u, pid); return jsonable(svc.store.get_events(pid))

    # ---------------------------------------------------------------- observations (ICU / sepsis) and disease assessments
    @app.post("/api/patients/{pid}/observations")
    def obs(pid: str, b: ObsIn, u=Depends(staff)):
        access(u, pid)
        r = guard(lambda: svc.add_observation(pid, b.values, b.timestamp, b.source)); svc.store.audit(u["username"], u["role"], "observation_added", pid); return r

    @app.post("/api/patients/{pid}/assessments/{disease}")
    def assess(pid: str, disease: str, b: AssessIn, u=Depends(current_user)):
        access(u, pid)
        r = guard(lambda: svc.assess_disease(pid, disease, b.values)); svc.store.audit(u["username"], u["role"], "assessment", pid, {"disease": disease}); return r

    @app.get("/api/patients/{pid}/assessments")
    def assessments(pid: str, disease: Optional[str] = None, u=Depends(current_user)):
        access(u, pid); return guard(lambda: svc.disease_risk_history(pid, disease))

    @app.post("/api/patients/{pid}/explain/{disease}")
    def explain(pid: str, disease: str, b: AssessIn, u=Depends(current_user)):
        access(u, pid)
        def f():
            m = svc.tabular_models.get(disease)
            if m is None or svc.registry[disease].status != "available": raise RuntimeError(f"{disease} model is not available")
            return m.explain(b.values)
        return guard(f)

    # ---------------------------------------------------------------- alerts + agent (human in the loop)
    @app.get("/api/alerts")
    def alerts(u=Depends(current_user)):
        al = svc.all_alerts()
        if u["role"] == "patient": al = [a for a in al if a["patient_id"] == u["patient_id"]]
        return jsonable(al)

    @app.post("/api/alerts/{alert_id}/action")
    def alert_action(alert_id: str, b: ActionIn, u=Depends(staff)):
        if b.action not in ("acknowledge", "escalate", "resolve"): raise HTTPException(400, "action must be acknowledge, escalate or resolve")
        r = guard(lambda: svc.alert_action(alert_id, b.action, b.note, u["username"])); svc.store.audit(u["username"], u["role"], f"alert_{b.action}", r.get("patient_id") if isinstance(r, dict) else None); return r

    @app.get("/api/agent/triage")
    def triage(u=Depends(staff)): return guard(lambda: over.triage())

    @app.post("/api/agent/run")
    def agent_run(u=Depends(staff)): return guard(lambda: over.run(by=u["username"]))

    @app.get("/api/agent/proposals")
    def proposals(status: Optional[str] = None, u=Depends(staff)): return jsonable(over.proposals(None, status))

    @app.post("/api/agent/proposals/{proposal_id}/decision")
    def decide(proposal_id: int, b: DecisionIn, u=Depends(staff)):
        return guard(lambda: over.decide(proposal_id, b.decision, u["username"], b.note))

    # ---------------------------------------------------------------- admin
    @app.get("/api/admin/users")
    def users(u=Depends(admin)): return [pub(x) for x in svc.store.list_users()]

    @app.get("/api/admin/audit")
    def audit(u=Depends(admin)): return jsonable(svc.store.get_audit())

    return app
