"""Patient-specific, evidence-based insights. Every insight states its source and the observation time it refers to.
Rule-based and local (no LLM). Only restates numbers the system computed."""
import numpy as np
from src.data_ingestion.sepsis_loader import VITALS
from src.data_ingestion.schema import SPECS

RANK = {"notable": 0, "attention": 1, "info": 2}
NAMES = {k: v.label for k, v in SPECS.items()}


def _time(hour, records):
    for r in records:
        if r["hour"] == hour:
            return r["timestamp"] or f"hour {hour}"
    return f"hour {hour}"


def _ins(cat, sev, text, source, time, evidence=None):
    return {"category": cat, "severity": sev, "text": text, "source": source, "time": time, "evidence": evidence or {}}


def generate_insights(snap: dict, profile: dict, occlusion: dict = None, category: str = None) -> list:
    if not snap.get("has_data"):
        return [_ins("data", "info", "No observations have been recorded for this patient yet.", "Digital Twin", "-")]
    s, recs, h = snap["state"], snap["obs_records"], snap["history_df"]
    hr = s["hour"]; now = _time(hr, recs); out = []
    # 1. deviation from the patient's own baseline
    for v in VITALS:
        b, cur = s["baseline"].get(v), s["latest_values"].get(v)
        if not b or cur is None:
            continue
        sd = max(b["baseline_sd"], 0.03 * abs(b["baseline_mean"]), 1e-6); z = (cur - b["baseline_mean"]) / sd
        if abs(z) >= 2:
            lh = hr - s["hours_since_measured"].get(v, 0)
            out.append(_ins("baseline", "notable" if abs(z) >= 3 else "attention",
                            f"{NAMES[v]} is {abs(cur - b['baseline_mean']):.1f} {SPECS[v].unit} {'above' if z > 0 else 'below'} this patient's own baseline "
                            f"({cur:.1f} vs {b['baseline_mean']:.1f}, {abs(z):.1f} SD).", f"Twin baseline = mean of first {b['n_baseline']} measured values; latest {v} measurement",
                            _time(lh, recs), {"variable": v, "value": cur, "baseline_mean": b["baseline_mean"], "z": z}))
    # 2. recent trends (measured points only)
    for v in VITALS:
        d = h[["hour", v]].dropna().tail(6)
        if len(d) < 4:
            continue
        slope = float(np.polyfit(d["hour"], d[v], 1)[0]); span = float(d["hour"].iloc[-1] - d["hour"].iloc[0])
        b = s["baseline"].get(v); floor = max(b["baseline_sd"], 0.03 * abs(b["baseline_mean"])) if b else max(float(d[v].std()), 1e-6)
        if span > 0 and abs(slope * span) >= 1.5 * floor:
            out.append(_ins("trend", "attention", f"{NAMES[v]} has {'risen' if slope > 0 else 'declined'} from {d[v].iloc[0]:.1f} to {d[v].iloc[-1]:.1f} "
                            f"across the last {len(d)} measurements.", f"Linear trend over measured {v} values, hours {int(d['hour'].iloc[0])}-{int(d['hour'].iloc[-1])}",
                            _time(int(d["hour"].iloc[-1]), recs), {"variable": v, "slope_per_hour": slope}))
    # 3. stale measurements
    for v in VITALS:
        age = s["hours_since_measured"].get(v)
        if age is not None and age >= 6:
            out.append(_ins("data_quality", "attention", f"{NAMES[v]} has not been measured for {age} hours; the model is using the last known value.",
                            "Digital Twin measurement ages", now, {"variable": v, "hours_since_measured": age}))
    # 4. model risk
    r = s["risk"]
    if r:
        rh = [x["risk_score"] for x in snap["risk_history"] if x.get("risk_score") is not None]
        txt = f"Sepsis model risk score is {r['risk_score']:.2f} (category: {category or 'n/a'}; model threshold {r['threshold']:.2f})."
        ev = {"score": r["risk_score"]}
        if len(rh) >= 7:
            txt += f" Six observations earlier it was {rh[-7]:.2f}."; ev["six_obs_ago"] = rh[-7]
        elif len(rh) >= 2:
            txt += f" At the previous observation it was {rh[-2]:.2f}."; ev["previous"] = rh[-2]
        out.append(_ins("risk", "notable" if r["flagged"] else "info", txt, f"Model {r['model_name']}, using observations up to hour {r['last_hour_used']}", now, ev))
    # 5. what the score leans on (occlusion)
    if occlusion and occlusion["items"]:
        top = [i for i in occlusion["items"] if abs(i["effect"]) >= 0.005][:3]
        if top:
            out.append(_ins("explanation", "info", "For this patient the score leans most on: " + "; ".join(
                f"{NAMES.get(i['feature'], i['feature'])} = {i['current_value']:.1f} ({i['effect']:+.2f} vs typical value)" for i in top) + ".",
                f"Occlusion analysis on {occlusion['model_name']} (not SHAP)", _time(occlusion["hour"], recs), {"items": top}))
    # 6. current anomaly detector output
    for d in s["deviations"][:3]:
        out.append(_ins("anomaly", "attention", d["explanation"], "Rolling per-patient baseline detector", now, {"variable": d["variable"], "z": d["z"]}))
    # 7. alerts
    al = snap["alerts"]
    if al:
        pend = [a for a in al if a["status"] in ("active", "escalated")]; last = al[-1]
        out.append(_ins("alerts", "attention" if pend else "info", f"{len(al)} alert(s) so far; {len(pend)} still need human review. Latest: {last['severity']} "
                        f"{last['type']} at {last.get('data_timestamp') or 'hour ' + str(last['hour'])} (status: {last['status']}).", "Alert history", now,
                        {"total": len(al), "pending": len(pend)}))
    # 8. patient profile context
    for k, rep in (profile.get("baseline") or {}).items():
        cur = s["latest_values"].get(k)
        if cur is not None and rep and abs(cur - rep) / abs(rep) >= 0.15:
            out.append(_ins("profile", "attention", f"{NAMES.get(k, k)} is {cur:.1f} versus the {rep:g} recorded as this patient's usual value in the profile.",
                            "Patient profile (baseline health information)", now, {"variable": k, "profile_value": rep, "current": cur}))
    unavailable = [c for c in profile.get("conditions", []) if c in ("Type 2 diabetes", "Cardiovascular disease", "Hypertension")]
    if unavailable:
        out.append(_ins("scope", "info", f"The profile lists {', '.join(unavailable)}; no validated prediction model exists for these in this system, so only sepsis risk is calculated.",
                        "Patient profile and disease registry", "-"))
    out.sort(key=lambda i: RANK[i["severity"]])
    return out
