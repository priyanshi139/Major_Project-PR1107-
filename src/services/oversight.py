"""Agentic alert layer bound to the patient platform: builds an evidence view per patient, ranks them with the deterministic TriageAgent and stores
PROPOSED follow-ups. Nothing executes: every proposal waits for a clinician's approve/reject decision (recorded with who and when)."""
from datetime import datetime
import numpy as np
from src.agents.triage import TriageAgent
from src.services.patient_service import SOURCE_LABELS, SYNTHETIC_SOURCES

DISEASES = ("diabetes", "cardiovascular", "hypertension")


def _kind(src):
    return "synthetic/simulated" if src in SYNTHETIC_SOURCES else SOURCE_LABELS.get(src, src or "unspecified")


def build_view(svc, pid) -> dict:
    sn = svc.snapshot(pid); p = sn["patient"]; recs = sn["obs_records"]; v = {"patient_id": pid, "name": p["name"], "synthetic": bool(p.get("synthetic", True)),
                                                                         "llm_consent": False, "sepsis": None, "chronic": {}, "hours_since_last_obs": None}
    pending = [a for a in sn["alerts"] if a["status"] in ("active", "escalated")]
    v["pending_alerts"] = [{"alert_id": a["alert_id"], "severity": a["severity"], "type": a["type"], "trigger": a["trigger"], "hour": a["hour"],
                            "data_timestamp": a.get("data_timestamp"), "data_source": a.get("data_source")} for a in pending]
    if sn["has_data"]:
        st = sn["state"]; r = st.get("risk") or {}; last = recs[-1]
        hist = [x["risk_score"] for x in sn["risk_history"] if x.get("risk_score") is not None][-6:]
        slope = float(np.polyfit(range(len(hist)), hist, 1)[0]) if len(hist) >= 3 else None
        v["sepsis"] = {"has_data": True, "score": r.get("risk_score"), "threshold": r.get("threshold"), "category": sn["category"], "model_name": r.get("model_name") or sn["model_name"],
                       "last_hour": st["hour"], "last_ts": last.get("timestamp"), "data_kind": _kind(last.get("source")), "slope": slope, "n_trend": len(hist),
                       "data_quality": st.get("data_quality")}
        if last.get("timestamp"):
            try:
                v["hours_since_last_obs"] = max(0.0, (datetime.now() - datetime.fromisoformat(last["timestamp"])).total_seconds() / 3600)
            except Exception:
                pass
    for d in DISEASES:
        rows = svc.store.get_disease_assessments(pid, d)
        if rows:
            cur, prev = rows[-1], (rows[-2] if len(rows) > 1 else None)
            v["chronic"][d] = {"name": cur.get("disease_name", d), "score": cur["risk_score"], "threshold": cur["threshold"], "category": cur.get("category") or cur.get("risk_category"),
                               "prev_score": prev["risk_score"] if prev else None, "model_name": cur.get("model_name", d), "timestamp": cur.get("assessed_at"), "data_kind": "patient assessment form"}
    return v


class Oversight:
    def __init__(self, svc, agent=None):
        self.svc, self.agent = svc, agent or TriageAgent(svc.cfg)

    def views(self, pids=None):
        return [build_view(self.svc, p["patient_id"]) for p in self.svc.list_patients() if pids is None or p["patient_id"] in pids]

    def triage(self, pids=None) -> list:
        return self.agent.rank(self.views(pids))

    def run(self, pids=None, by="agent") -> dict:
        """Rank patients and store NEW proposals only (an open proposal for the same patient+disease is not duplicated)."""
        views = {v["patient_id"]: v for v in self.views(pids)}
        ranking = sorted((self.agent.priority(v) for v in views.values()), key=lambda o: -o["priority"])
        for o in ranking:
            o["narrative"] = self.agent.narrate(o)
        store, new = self.svc.store, 0
        for entry in ranking:
            pid = entry["patient_id"]
            open_ = {(q["disease"]) for q in store.get_proposals(pid, "proposed")}
            for pr in self.agent.propose(entry, views[pid]):
                if pr["disease"] in open_:
                    continue
                store.add_proposal(pid, (entry["pending_alert_ids"] or [None])[0], pr["disease"], pr["title"], pr["rationale"], pr["evidence"]); new += 1
        store.audit(by, None, "agent_run", None, {"patients": len(ranking), "new_proposals": new})
        return {"ranking": ranking, "new_proposals": new, "note": "Proposals are suggestions for a clinician; none has been acted on."}

    def proposals(self, pid=None, status=None):
        return self.svc.store.get_proposals(pid, status)

    def decide(self, proposal_id, decision, by, note=""):
        self.svc.store.decide_proposal(proposal_id, decision, by, note)
        self.svc.store.audit(by, None, f"proposal_{decision}", None, {"proposal_id": proposal_id, "note": note})
        return next(p for p in self.svc.store.get_proposals() if p["proposal_id"] == proposal_id)
