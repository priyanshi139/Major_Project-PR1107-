"""Agentic alert layer: offline-first, deterministic, evidence-grounded, human-in-the-loop.

The agent is a small tool-using planner. For every patient it calls fixed TOOLS (latest risk, trend, alerts, data quality, staleness),
collects EVIDENCE items (each with value, source, time, data kind), scores a transparent priority, groups duplicate alerts into
incidents, and drafts PROPOSED follow-up steps. It never acts: every proposal waits for a clinician's approve/reject decision.
An LLM is optional (HEALTHTWIN_LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY): it may only reword the template text, and its output is
discarded unless every number in it already appears in the evidence (grounding check)."""
import json, os, re
import numpy as np

CRITICAL_W, WARNING_W, INFO_W = 15.0, 6.0, 1.0
PRIORITY_TIERS = [(60.0, "urgent review"), (35.0, "review soon"), (0.0, "routine")]

PROPOSALS = {   # templates; phrased as options for clinician decision, never orders
    "sepsis": ("Request bedside clinician review of possible sepsis",
               "The sepsis model score reached its alert threshold. Suggest a clinician reviews the patient and decides on a full set of vital signs "
               "and sepsis-screening investigations per local protocol."),
    "diabetes": ("Suggest confirmatory diabetes testing and clinician review",
                 "The diabetes screening model flagged elevated risk. Suggest a clinician considers confirmatory testing (e.g. HbA1c or fasting glucose)."),
    "cardiovascular": ("Suggest clinician review of cardiovascular risk factors",
                       "The 10-year CHD model flagged elevated risk. Suggest a clinician reviews blood pressure, lipids and lifestyle factors."),
    "hypertension": ("Suggest repeat blood-pressure measurement",
                     "The hypertension screening model flagged elevated likelihood. Suggest repeat or ambulatory blood-pressure measurement by a clinician."),
}


def ev(claim, value, source, time, kind):
    return {"claim": claim, "value": value, "source": source, "time": time, "data_kind": kind}


class TriageAgent:
    def __init__(self, cfg=None, llm=None):
        c = (cfg or {}).get("agent", {}) if cfg else {}
        self.w = {"sepsis": c.get("w_sepsis", 40.0), "chronic": c.get("w_chronic", 20.0), "trend": c.get("w_trend", 10.0),
                  "alerts": c.get("w_alerts_cap", 30.0), "quality": c.get("w_quality", 5.0)}
        self.stale_hours = c.get("stale_hours", 12)
        self.llm = llm if llm is not None else _llm_from_env()

    # ------------------------------------------------------------------ tools
    def tool_sepsis(self, v):
        s = v.get("sepsis") or {}
        if not s.get("has_data"):
            return None, []
        e = []
        if s.get("score") is not None:
            e.append(ev(f"Sepsis model risk score {s['score']:.2f} vs threshold {s['threshold']:.2f} ({s['category']})", s["score"],
                        f"{s['model_name']} using data up to hour {s['last_hour']}", s.get("last_ts") or f"hour {s['last_hour']}", s["data_kind"]))
        if s.get("slope") is not None and abs(s["slope"]) >= 0.01:
            e.append(ev(f"Sepsis score trend {s['slope']:+.3f} per observation over the last {s['n_trend']} scores", s["slope"],
                        "linear fit of the patient's own risk history", s.get("last_ts") or f"hour {s['last_hour']}", s["data_kind"]))
        return s, e

    def tool_chronic(self, v):
        out, e = {}, []
        for d, r in (v.get("chronic") or {}).items():
            out[d] = r
            e.append(ev(f"{r['name']} risk {r['score']:.2f} vs threshold {r['threshold']:.2f} ({r['category']})"
                        + (f"; previous check-in {r['prev_score']:.2f}" if r.get("prev_score") is not None else ""), r["score"],
                        f"{r['model_name']} on check-in data", r.get("timestamp") or "latest check-in", r["data_kind"]))
        return out, e

    def tool_alerts(self, v):
        pend = v.get("pending_alerts") or []
        return pend, [ev(f"{a['severity']} alert ({a['type']}) awaiting review: {a['trigger']}", a["severity"], "alert manager",
                         a.get("data_timestamp") or f"hour {a['hour']}", a.get("data_source") or "unspecified") for a in pend[:6]]

    # ------------------------------------------------------------------ priority
    def priority(self, v: dict) -> dict:
        ev_all, comp, reasons = [], {}, []
        s, e1 = self.tool_sepsis(v); ev_all += e1
        sep = 0.0
        if s and s.get("score") is not None and s.get("threshold"):
            sep = float(np.clip(s["score"] / s["threshold"], 0, 2)) / 2 * self.w["sepsis"]
            if s["score"] >= s["threshold"]:
                reasons.append(f"sepsis score {s['score']:.2f} is at or above its threshold {s['threshold']:.2f}")
        comp["sepsis_risk"] = sep
        ch, e2 = self.tool_chronic(v); ev_all += e2
        cr = 0.0
        for d, r in ch.items():
            ratio = float(np.clip(r["score"] / max(r["threshold"], 1e-6), 0, 2)) / 2
            cr = max(cr, ratio * self.w["chronic"])
            if r["score"] >= r["threshold"]:
                reasons.append(f"{r['name']} risk {r['score']:.2f} is at or above its threshold {r['threshold']:.2f}")
        comp["chronic_risk"] = cr
        tr = 0.0
        if s and s.get("slope") is not None and s["slope"] > 0:
            tr = float(np.clip(s["slope"] / 0.05, 0, 1)) * self.w["trend"]
            if s["slope"] >= 0.02:
                reasons.append(f"sepsis score is rising ({s['slope']:+.3f} per observation)")
        comp["trend"] = tr
        pend, e3 = self.tool_alerts(v); ev_all += e3
        al = min(self.w["alerts"], sum({"critical": CRITICAL_W, "warning": WARNING_W}.get(a["severity"], INFO_W) for a in pend))
        if pend:
            c = {k: sum(1 for a in pend if a["severity"] == k) for k in ("critical", "warning", "info")}
            reasons.append(f"{len(pend)} alert(s) awaiting human review ({c['critical']} critical, {c['warning']} warning, {c['info']} info)")
        comp["unreviewed_alerts"] = al
        q = 0.0
        dq = (s or {}).get("data_quality")
        stale = v.get("hours_since_last_obs")
        if dq == "poor":
            q += self.w["quality"] * 0.6; reasons.append("data quality of the latest observation is poor, so the model has less to go on")
        if stale is not None and stale >= self.stale_hours:
            q += self.w["quality"] * 0.4; reasons.append(f"no new observation for {stale:.0f} h")
        comp["data_uncertainty"] = q
        score = float(np.clip(sum(comp.values()), 0, 100))
        tier = next(t for th, t in PRIORITY_TIERS if score >= th)
        return {"patient_id": v["patient_id"], "name": v.get("name"), "synthetic": v.get("synthetic", True), "priority": score, "tier": tier,
                "components": comp, "llm_consent": bool(v.get("llm_consent")), "reasons": reasons, "evidence": ev_all, "pending_alert_ids": [a["alert_id"] for a in pend]}

    def rank(self, views: list) -> list:
        out = [self.priority(v) for v in views]
        for o in out:
            o["narrative"] = self.narrate(o)
        return sorted(out, key=lambda o: -o["priority"])

    # ------------------------------------------------------------------ duplicate reduction
    @staticmethod
    def group_alerts(alerts: list) -> list:
        """Collapse repeated alerts of the same patient/type/severity/domain into one incident with a count and time span."""
        groups = {}
        for a in alerts:
            groups.setdefault((a["patient_id"], a["type"], a["severity"], a.get("domain")), []).append(a)
        inc = []
        for (pid, typ, sev, dom), g in groups.items():
            g = sorted(g, key=lambda a: a["hour"])
            inc.append({"patient_id": pid, "type": typ, "severity": sev, "domain": dom, "count": len(g), "first_hour": g[0]["hour"],
                        "last_hour": g[-1]["hour"], "alert_ids": [a["alert_id"] for a in g], "latest_alert_id": g[-1]["alert_id"],
                        "open": sum(1 for a in g if a["status"] in ("active", "escalated")), "latest_status": g[-1]["status"],
                        "latest_explanation": g[-1]["explanation"], "max_score": max([a["score"] for a in g if a.get("score") is not None], default=None)})
        order = {"critical": 0, "warning": 1, "info": 2}
        return sorted(inc, key=lambda i: (order.get(i["severity"], 3), -i["open"], -i["last_hour"]))

    # ------------------------------------------------------------------ proposals (never executed)
    @staticmethod
    def propose(entry: dict, v: dict) -> list:
        props = []
        s = v.get("sepsis") or {}
        if s.get("has_data") and s.get("score") is not None and s["score"] >= s["threshold"]:
            t, r = PROPOSALS["sepsis"]
            props.append({"disease": "sepsis", "title": t, "rationale": r,
                          "evidence": [e for e in entry["evidence"] if "Sepsis" in e["claim"]][:3]})
        for d, c in (v.get("chronic") or {}).items():
            if c["score"] >= c["threshold"] and d in PROPOSALS:
                t, r = PROPOSALS[d]
                props.append({"disease": d, "title": t, "rationale": r, "evidence": [e for e in entry["evidence"] if c["name"] in e["claim"]][:2]})
        return props

    # ------------------------------------------------------------------ narrative (template; optional grounded LLM rewrite)
    def narrate(self, o: dict) -> str:
        if o["reasons"]:
            text = (f"{o['name'] or o['patient_id']}: priority {o['priority']:.0f}/100 ({o['tier']}). Because " + "; ".join(o["reasons"][:4]) + ". "
                    "These are model-generated flags for clinician review, not diagnoses.")
        else:
            text = f"{o['name'] or o['patient_id']}: priority {o['priority']:.0f}/100 ({o['tier']}). No risk threshold is currently exceeded."
        if self.llm and o.get("llm_consent"):
            try:
                alt = self.llm(text, o["evidence"])
                if alt and grounded(alt, text, o["evidence"]):
                    return alt
            except Exception:
                pass
        return text


_NUM = re.compile(r"\d+(?:\.\d+)?")


def grounded(candidate: str, base_text: str, evidence: list) -> bool:
    """Every number in an LLM rewrite must already occur in the template text or the evidence (no invented figures)."""
    allowed = set(_NUM.findall(base_text + " " + json.dumps(evidence, default=str)))
    allowed |= {str(int(float(x))) for x in list(allowed) if float(x) == int(float(x))}
    return all(n in allowed for n in _NUM.findall(candidate))


def _llm_from_env():
    """Optional LLM reword. Off by default. Needs HEALTHTWIN_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY (not exercised in this repo's tests)."""
    if os.environ.get("HEALTHTWIN_LLM_PROVIDER", "").lower() != "anthropic" or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import requests
    model = os.environ.get("HEALTHTWIN_LLM_MODEL", "claude-haiku-4-5")

    def call(text, evidence):
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=30,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model, "max_tokens": 300,
                                "system": "Rewrite the clinical triage note in plain, calm language for a clinician. Use ONLY facts and numbers present in the note. "
                                          "Do not add diagnoses, treatments or new numbers. Keep it under 80 words.",
                                "messages": [{"role": "user", "content": text}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    return call
