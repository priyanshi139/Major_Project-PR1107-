import time, uuid

PROTOTYPE_RULES = (
    "Alert rules: 'warning' = model score reaches the model threshold (set from validation data); "
    "'critical' = score reaches the critical level (configurable, default 0.90); "
    "'info' = a reading unusual versus the patient's own baseline. Alerts are model-generated warnings for human review. "
    "Escalation only changes an alert's review status; the system performs no clinical action.")

STATUSES = ("active", "acknowledged", "escalated", "resolved")
_ALLOWED = {"active": {"acknowledged", "escalated", "resolved"},
            "acknowledged": {"escalated", "resolved"},
            "escalated": {"acknowledged", "resolved"},
            "resolved": set()}


def _event(status, by, note=""):
    return {"status": status, "by": by, "note": note, "wall_time": time.time()}


class AlertManager:
    def __init__(self, cfg=None, db=None):
        cfg = cfg or {}
        self.critical_threshold = cfg.get("critical_threshold", 0.90)
        self.warning_threshold = cfg.get("warning_threshold")      # None -> use model's own threshold
        self.cooldown_hours = cfg.get("cooldown_hours", 6)
        self.escalate_after_hours = cfg.get("escalate_after_hours", 3)   # unacknowledged critical alerts
        self.risk_rise_delta = cfg.get("risk_rise_delta", 0.20)         # model-trend rule: score rise within the window
        self.risk_rise_window = cfg.get("risk_rise_window", 3)
        self.risk_rise_min_score = cfg.get("risk_rise_min_score", 0.25)
        self.multi_vital_min = cfg.get("multi_vital_min", 2)            # vitals unusual at the same time
        self.db = db
        self._alerts, self._last_fired, self.suppressed = [], {}, 0

    def _emit(self, subject_id, module, hour, typ, severity, score, factors, explanation, domain="sepsis model", trigger="", context=None):
        key = (subject_id, typ, severity)
        last = self._last_fired.get(key)
        if last is not None and 0 <= hour - last < self.cooldown_hours:
            self.suppressed += 1
            return None                                            # duplicate suppression (cooldown)
        self._last_fired[key] = hour
        a = {"alert_id": "ALERT-" + uuid.uuid4().hex[:8], "subject_id": subject_id, "module": module,
             "hour": hour, "type": typ, "severity": severity, "score": score, "factors": factors,
             "explanation": explanation, "status": "active", "model_generated": True,
             "created_wall_time": time.time(), "acknowledged_by": None, "acknowledged_wall_time": None,
             "escalated_to": None, "history": [_event("active", "system", "created by rule")],
             "patient_id": subject_id.split(":", 1)[-1], "domain": domain, "trigger": trigger,
             "relevant_observations": (context or {}).get("values", {}), "data_timestamp": (context or {}).get("timestamp"),
             "data_source": (context or {}).get("source")}
        self._alerts.append(a)
        self._save(a)
        return a

    def _save(self, a):
        if self.db:
            self.db.save_alert(a)

    def evaluate(self, subject_id, module, hour, risk=None, anomalies=None, risk_history=None, context=None) -> list:
        """Rules: model score >= threshold (warning/critical); score rising quickly (risk_rise); several vitals unusual at once
        (multi_vital); a single vital unusual vs own baseline (anomaly, info). All thresholds are configurable."""
        new = []
        if risk:
            thr = self.warning_threshold if self.warning_threshold is not None else risk["threshold"]
            s = risk["risk_score"]
            if s >= thr:
                sev = "critical" if s >= self.critical_threshold else "warning"
                factors = risk.get("top_factors") or []
                ftxt = ("; top contributing features (association, not causation): " +
                        ", ".join(f"{f['feature']} ({f['contribution']:+.2f})" for f in factors)) if factors else ""
                a = self._emit(subject_id, module, hour, "risk", sev, s, factors,
                               f"Model-generated warning (not a diagnosis): risk score {s:.2f} reached threshold "
                               f"{thr:.2f} at hour {hour} using {risk['n_observations_used']} observations "
                               f"(hours 0-{risk['last_hour_used']}){ftxt}.", "sepsis model", f"model score {s:.2f} >= threshold {thr:.2f}", context)
                if a: new.append(a)
            elif risk_history and len(risk_history) > self.risk_rise_window:
                base = risk_history[-(self.risk_rise_window + 1)]
                if s - base >= self.risk_rise_delta and s >= self.risk_rise_min_score:
                    a = self._emit(subject_id, module, hour, "risk_rise", "warning", s, [],
                                   f"Model risk score rose from {base:.2f} to {s:.2f} over the last {self.risk_rise_window} observations "
                                   f"(still below the model threshold {thr:.2f}).", "sepsis model",
                                   f"score rise {s - base:+.2f} within {self.risk_rise_window} observations (limit {self.risk_rise_delta:.2f})", context)
                    if a: new.append(a)
        if anomalies:
            top = anomalies[:3]; vars_ = sorted({d["variable"] for d in anomalies})
            fac = [{"feature": d["variable"], "z": d["z"]} for d in top]
            if len(vars_) >= self.multi_vital_min:
                a = self._emit(subject_id, module, hour, "multi_vital", "warning", None, fac,
                               f"{len(vars_)} vital signs are unusual versus this patient's own baseline at the same time ({', '.join(vars_)}). "
                               + " ".join(d["explanation"] for d in top), "vital signs",
                               f">= {self.multi_vital_min} vitals beyond the baseline limit simultaneously", context)
            else:
                a = self._emit(subject_id, module, hour, "anomaly", "info", None, fac, " ".join(d["explanation"] for d in top),
                               "vital signs", f"{vars_[0]} beyond the baseline limit", context)
            if a: new.append(a)
        self.auto_escalate(subject_id, hour)
        return new

    def annotate(self, alert_id, **fields):
        a = self._get(alert_id); a.update(fields); self._save(a); return a

    def restore(self, alerts: list):
        """Reload persisted alerts (e.g. after an app restart) so history, cooldowns and review status continue."""
        have = {a["alert_id"] for a in self._alerts}
        for a in alerts:
            if a["alert_id"] not in have:
                self._alerts.append(a)
            key = (a["subject_id"], a["type"], a["severity"])
            self._last_fired[key] = max(self._last_fired.get(key, -1), a["hour"])

    # ---- human-oversight workflow ----
    def _get(self, alert_id):
        for a in self._alerts:
            if a["alert_id"] == alert_id:
                return a
        raise KeyError(f"Unknown alert id {alert_id}")

    def _transition(self, alert_id, status, by, note="", **extra):
        a = self._get(alert_id)
        if status not in _ALLOWED[a["status"]]:
            raise ValueError(f"Cannot change alert from '{a['status']}' to '{status}'")
        a["status"] = status
        a["history"].append(_event(status, by, note))
        a.update(extra)
        self._save(a)
        return a

    def acknowledge(self, alert_id, by="clinician", note=""):
        return self._transition(alert_id, "acknowledged", by, note, acknowledged_by=by,
                                acknowledged_wall_time=time.time())

    def escalate(self, alert_id, to="senior_clinician", by="clinician", note=""):
        return self._transition(alert_id, "escalated", by, note, escalated_to=to)

    def resolve(self, alert_id, by="clinician", note=""):
        return self._transition(alert_id, "resolved", by, note)

    def auto_escalate(self, subject_id, hour):
        """Critical alerts nobody acknowledged within N simulated hours are marked 'escalated' for human review."""
        out = []
        for a in self._alerts:
            if (a["subject_id"] == subject_id and a["severity"] == "critical" and a["status"] == "active"
                    and hour - a["hour"] >= self.escalate_after_hours):
                out.append(self._transition(a["alert_id"], "escalated", "system rule",
                                            f"no acknowledgement within {self.escalate_after_hours} h",
                                            escalated_to="on-call clinician (simulated; no real notification sent)"))
        return out

    def active(self, subject_id=None):
        return [a for a in self._alerts if a["status"] == "active" and (not subject_id or a["subject_id"] == subject_id)]

    def pending(self, subject_id=None):
        """Alerts still needing human review (active or escalated)."""
        return [a for a in self._alerts if a["status"] in ("active", "escalated")
                and (not subject_id or a["subject_id"] == subject_id)]

    def history(self, subject_id=None):
        return [a for a in self._alerts if not subject_id or a["subject_id"] == subject_id]

    def reset_subject(self, subject_id):
        self._last_fired = {k: v for k, v in self._last_fired.items() if k[0] != subject_id}
        self._alerts = [a for a in self._alerts if a["subject_id"] != subject_id]
