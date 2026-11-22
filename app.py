"""HealthTwin - patient-centric Digital Twin platform.  Run:  streamlit run app.py
Registration -> profile -> twin -> observations -> twin update -> prediction -> insights -> alerts -> timeline -> what-if.
Data labels: public dataset records, SYNTHETIC demo data, SIMULATED live data, manual entries, uploads. Nothing here is a live hospital feed."""
import io, json, os, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.utils.config import load_config, resolve_path
from src.data_ingestion.sepsis_loader import VITALS, LABS
from src.data_ingestion.schema import SPECS
from src.data_ingestion.upload import validate_patient_upload, template_csv
from src.digital_twin.whatif import ADJUSTABLE
from src.insights.narrative import explain_whatif
from src.simulation.session import load_assets
from src.simulation.feed import SCENARIOS
from src.services.patient_service import PatientService, CONDITIONS, GENDERS, ICU_UNITS, SOURCE_LABELS, SYNTHETIC_SOURCES
from src.services.oversight import Oversight
from src.alerts.manager import PROTOTYPE_RULES
from src.ui import theme

st.set_page_config(page_title="HealthTwin", page_icon="🩺", layout="wide")
theme.inject_css()
cfg = load_config(os.environ.get("HEALTHTWIN_CONFIG"))
DISCLAIMER = cfg["project"]["disclaimer"]


@st.cache_resource(show_spinner="Loading data and models ...")
def get_assets(cfg_key):
    return load_assets(cfg)


assets = get_assets(os.environ.get("HEALTHTWIN_CONFIG", "default"))
if "svc" not in st.session_state:
    st.session_state["svc"] = PatientService(cfg, assets)
svc: PatientService = st.session_state["svc"]
_dp = assets.get("data_provenance")
if _dp and _dp.get("data_origin") not in ("real_public", "user_confirmed_real"):
    st.warning(f"Processed dataset origin: **{_dp.get('data_origin')}** ({_dp.get('source')}). Historical replay and any model trained on it must NOT be presented as real PhysioNet results. "
               "If these are the real files, run: python scripts/prepare_sepsis_data.py --confirm-physionet-origin")

P_OV, P_PATS, P_PROF, P_TWIN, P_LIVE, P_PRED, P_TIME, P_WHAT, P_INS, P_ALERT, P_DATA = (
    "🏠 Overview", "👥 Patients", "🪪 Profile", "🧬 Twin", "📡 Live", "🔬 Prediction", "🕒 Timeline", "🔮 What-if", "💡 Insights", "🚨 Alerts", "🗄 Data")
PAGE_LIST = [P_OV, P_PATS, P_PROF, P_TWIN, P_LIVE, P_PRED, P_TIME, P_WHAT, P_INS, P_ALERT, P_DATA]
SEV_RANK = {"critical": 0, "warning": 1, "info": 2}


# ---------------- callbacks (run before the next render) ----------------
def go_to(page_name): st.session_state["page"] = page_name
def flash(kind, msg): st.session_state.setdefault("_flash", []).append((kind, msg))


def select_patient(pid_, page_name=P_TWIN):
    st.session_state["pid_sel"] = pid_; go_to(page_name)


def cb_demo():
    try:
        ids = svc.create_demo_patients(); flash("success", f"Created {len(ids)} SYNTHETIC demo patients: {', '.join(ids)}"); select_patient(ids[1], P_OV)
    except Exception as e: flash("error", str(e))


def cb_register():
    g = st.session_state
    base = {k: g.get(f"reg_b_{k}") for k in ("HR", "MAP", "SBP", "Resp", "O2Sat", "Temp")}
    try:
        pid_ = svc.register_patient({"name": g.get("reg_name"), "age": g.get("reg_age"), "gender": g.get("reg_gender"), "conditions": g.get("reg_conditions"),
                                     "history": g.get("reg_history"), "baseline": base, "icu_unit": g.get("reg_unit"), "synthetic": g.get("reg_synth", True),
                                     "monitoring_start": datetime.combine(g["reg_date"], g["reg_time"]).isoformat()})
        flash("success", f"Patient {pid_} registered and its Digital Twin created."); select_patient(pid_, P_PROF)
    except Exception as e: flash("error", str(e))


def cb_save_profile(pid_):
    g = st.session_state
    try:
        svc.update_patient(pid_, {"name": g.get("prof_name"), "age": g.get("prof_age"), "gender": g.get("prof_gender"), "conditions": g.get("prof_conditions"),
                                  "history": g.get("prof_history"), "icu_unit": g.get("prof_unit"), "synthetic": g.get("prof_synth"),
                                  "baseline": {k: g.get(f"prof_b_{k}") for k in ("HR", "MAP", "SBP", "Resp", "O2Sat", "Temp")}})
        flash("success", "Profile saved.")
    except Exception as e: flash("error", str(e))


def cb_delete(pid_):
    if st.session_state.get("del_confirm"):
        svc.delete_patient(pid_); st.session_state["pid_sel"] = None; flash("success", f"Patient {pid_} deleted."); go_to(P_PATS)
    else: flash("error", "Tick the confirmation box first.")


def cb_save_obs(pid_):
    g = st.session_state; n = g.get("obs_n", 0)
    try:
        ts = datetime.combine(g[f"obs_date_{n}"], g[f"obs_time_{n}"])
        res = svc.add_observation(pid_, {k: g.get(f"obs_{k}_{n}") for k in SPECS}, ts, "manual_entry")
        g["_last_obs"] = {"pid": pid_, "hour": res["hour"], "ts": res["timestamp"], "risk": res["risk"], "status": res["state"]["status"],
                          "alerts": [(a["severity"], a["trigger"]) for a in res["new_alerts"]], "note": res["slot_note"]}
        g["obs_n"] = n + 1
    except Exception as e: flash("error", str(e)); g.pop("_last_obs", None)


def cb_start_feed(pid_):
    g = st.session_state
    try: svc.start_simulated_feed(pid_, g.get("feed_scenario", "stable"), int(g.get("feed_hours", 12)), float(g.get("feed_interval", 1.0)))
    except Exception as e: flash("error", str(e))


def cb_alert_action():
    g = st.session_state
    try: svc.alert_action(g["alert_pick"], g["alert_action"].lower(), g.get("alert_note", "")); flash("success", f"Alert {g['alert_pick']} updated.")
    except Exception as e: flash("error", str(e))


def cb_upload_patient(pid_):
    try: n = svc.append_uploaded(pid_, st.session_state["upload_clean"]); flash("success", f"{n} observations added to {pid_}.")
    except Exception as e: flash("error", str(e))


def cb_import_hist():
    g = st.session_state
    try:
        pid_ = svc.import_historical(g["imp_record"], int(g.get("imp_n", 12))); flash("success", f"Imported as {pid_}."); select_patient(pid_, P_LIVE)
    except Exception as e: flash("error", str(e))


# ---------------- helpers ----------------
def fmt_time(rec): return rec["timestamp"].replace("T", " ") if rec["timestamp"] else f"hour {rec['hour']} (relative)"


def patient_badges(p):
    if p["synthetic"]: return theme.badge("SYNTHETIC / DEMO PATIENT", "warn")
    if p["origin"] == "physionet_2019_record": return theme.badge("PUBLIC DATASET RECORD", "info")
    return theme.badge("USER-ENTERED PATIENT", "ok")


def label_of(p): return f"{p['patient_id']} · {p['name']}" + ("  [SYNTHETIC]" if p["synthetic"] else "")


def show_flash():
    for kind, msg in st.session_state.pop("_flash", []):
        {"success": st.success, "error": st.error, "info": st.info}.get(kind, st.info)(msg)


def head(title, subtitle, sn=None, p=None):
    p = p or (sn["patient"] if sn else None)
    theme.page_header(title, subtitle, (patient_badges(p) + theme.badge(p["patient_id"], "neutral")) if p else "")


def page_start(title, subtitle):
    """Flash messages + a single header (with patient badges when a patient is active). Returns (snapshot, stop)."""
    show_flash()
    sn = svc.snapshot(pid) if pid else None
    head(title, subtitle, sn)
    if not pid:
        need_patient(); return None, True
    return sn, False


def need_patient(tag=""):
    st.info("No patient selected. Register one, or create synthetic demo patients to explore the platform.")
    c1, c2 = st.columns(2)
    c1.button("Create 3 synthetic demo patients", key=f"demo_{st.session_state['page']}{tag}", on_click=cb_demo)
    c2.button("Register a patient", key=f"toreg_{st.session_state['page']}{tag}", on_click=go_to, args=(P_PATS,))


def need_data(sn):
    if sn["has_data"]: return False
    st.info("This patient has no observations yet. Add one on the **Live** page.")
    st.button("Go to Live monitoring", key=f"tolive_{st.session_state['page']}", on_click=go_to, args=(P_LIVE,))
    return True


def show(fig, container=st):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151"))
    container.plotly_chart(fig, width="stretch")


SRC_STYLE = {"manual_entry": ("Observed (manual entry)", "#2563eb", "circle"), "uploaded": ("Observed (uploaded)", "#0891b2", "circle"),
             "historical_dataset": ("Public dataset record", "#4f46e5", "circle"), "historical_replay": ("Public dataset (replayed)", "#7c3aed", "circle-open"),
             "synthetic_demo": ("SYNTHETIC demo", "#d97706", "square"), "simulated_live": ("SIMULATED live", "#ea580c", "triangle-up"),
             "unspecified": ("Observed", "#64748b", "circle")}


def trend_fig(sn, var, horizon_fc=True):
    h, recs = sn["history_df"], sn["obs_records"]
    src = {r["hour"]: r for r in recs}
    d = h[["hour", var]].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["hour"], y=d[var], mode="lines", line=dict(color="#cbd5e1", width=2), hoverinfo="skip", showlegend=False))
    for s_, (nm, col, sym) in SRC_STYLE.items():
        dd = d[d["hour"].map(lambda x: src.get(x, {}).get("source") == s_)]
        if len(dd):
            fig.add_trace(go.Scatter(x=dd["hour"], y=dd[var], mode="markers", name=nm, marker=dict(color=col, symbol=sym, size=8),
                                     text=[fmt_time(src[x]) for x in dd["hour"]], hovertemplate="%{text}<br>%{y}<extra>" + nm + "</extra>"))
    if len(d):
        fig.add_trace(go.Scatter(x=[d["hour"].iloc[-1]], y=[d[var].iloc[-1]], mode="markers", name="Current (latest measurement)",
                                 marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="#111827", width=2))))
    b = sn["state"]["baseline"].get(var)
    if b: fig.add_hline(y=b["baseline_mean"], line_dash="dash", line_color="#94a3b8", annotation_text="patient baseline")
    if horizon_fc and var in VITALS and assets.get("forecaster") is not None:
        for hz in assets["forecaster"].horizons:
            r = svc.forecast(sn["patient_id"], var, hz)
            if r.get("available"):
                fig.add_trace(go.Scatter(x=[r["target_hour"]], y=[r["value"]], mode="markers", name=f"FORECAST +{hz}h (statistical, not observed)",
                                         marker=dict(size=11, symbol="diamond", color="#ec4899"),
                                         error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["value"]], arrayminus=[r["value"] - r["lo"]])))
    fig.update_layout(height=360, xaxis_title="observation hour (relative to monitoring start)", yaxis_title=f"{SPECS[var].label} ({SPECS[var].unit})",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def risk_fig(sn):
    rh = pd.DataFrame(sn["risk_history"])
    fig = go.Figure(go.Scatter(x=rh["hour"], y=rh["risk_score"], mode="lines+markers", name="model risk score", line=dict(color="#7c3aed", width=3)))
    fig.add_hline(y=sn["model_threshold"], line_dash="dash", line_color="#f59e0b", annotation_text="model threshold")
    fig.add_hline(y=sn["critical_threshold"], line_dash="dot", line_color="#ef4444", annotation_text="critical level")
    fig.update_layout(yaxis=dict(range=[-0.05, 1.05], title="risk score"), xaxis_title="observation hour", height=340, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def tone(sn):
    r = sn["state"]["risk"] if sn["has_data"] else {}
    return "neutral" if not r else {"Critical": "crit", "High": "warn", "Elevated": "info"}.get(sn["category"], "ok")


# ---------------- sidebar ----------------
patients = svc.list_patients()
ids = [p["patient_id"] for p in patients]
if st.session_state.get("pid_sel") not in ids:
    st.session_state["pid_sel"] = ids[0] if ids else None
sb = st.sidebar
sb.markdown('<div class="ht-brand">🩺 HealthTwin</div>', unsafe_allow_html=True)
pid = None
if ids:
    pid = sb.selectbox("Active patient", ids, key="pid_sel", format_func=lambda i: label_of(next(p for p in patients if p["patient_id"] == i)))
    sn0 = svc.snapshot(pid); p0 = sn0["patient"]
    with sb:
        theme.sidebar_patient(pid[-3:], p0["name"], f"{p0['age']} y · {p0['gender']}" + (" · SYNTHETIC" if p0["synthetic"] else ""))
    sb.divider(); sb.markdown("#### 📊 Quick stats")
    if sn0["has_data"]:
        s0, r0 = sn0["state"], sn0["state"]["risk"]
        lines = [f"🎯 Risk: <b>{r0['risk_score']:.2f} ({sn0['category']})</b>" if r0 else "🎯 Risk: <b>no model</b>", f"🧬 Status: <b>{s0['status'].replace('_', ' ')}</b>",
                 f"🕒 Observations: <b>{s0['n_observations']}</b>", f"🚨 Needs review: <b>{sn0['pending_alerts']}</b>"]
        sb.markdown('<div class="ht-qs">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    else:
        sb.caption("No observations yet.")
    fs = svc.feed_status(pid)
    if fs: sb.markdown(f'<div class="ht-pill-status">{"🟢" if fs["status"] == "running" else "⏸"} {fs["label"].upper()}: {fs["status"]} ({fs["position"]}/{fs["total"]})</div>', unsafe_allow_html=True)
else:
    sb.caption("No patients yet. Create demo patients on the Overview page or register one under Patients.")
sb.divider()
sb.caption("Replay and simulated feeds are controlled on the Live page.")

with st.container(key="topnav"):
    page = st.radio("Navigate", PAGE_LIST, key="page", horizontal=True, label_visibility="collapsed")


# ================= pages: overview, patients, profile =================
def page_overview():
    theme.page_header("Overview", "Cohort status, alerts and recent activity"); show_flash()
    sums = svc.summaries()
    if not sums:
        theme.hero("Welcome to HealthTwin", "Register a patient to create their Digital Twin, then add observations to see risk, insights, alerts and a full timeline.")
        need_patient(); return
    theme.cards([("Patients", len(sums), f"{sum(s['synthetic'] for s in sums)} synthetic / demo", "purple"),
                 ("Flagged by model now", sum(s["flagged"] for s in sums), "warning for human review", "crit" if any(s["flagged"] for s in sums) else "ok"),
                 ("Alerts needing review", sum(s["pending_alerts"] for s in sums), "active or escalated", "warn" if any(s["pending_alerts"] for s in sums) else "ok"),
                 ("Observations recorded", sum(s["n_obs"] for s in sums), "all patients", "info")])
    rows = [{"Patient": s["patient_id"], "Name": s["name"], "Data": "SYNTHETIC" if s["synthetic"] else ("Public record" if s["origin"] == "physionet_2019_record" else "User-entered"),
             "Conditions": ", ".join(s["conditions"]) or "-", "Twin status": s["status"].replace("_", " "),
             "Risk": None if s["risk_score"] is None else round(s["risk_score"], 2), "Category": s["category"] or "-", "Observations": s["n_obs"],
             "Last update": (s["last_update"] or "-").replace("T", " "), "Needs review": s["pending_alerts"]} for s in sorted(sums, key=lambda s: (not s["flagged"], s["patient_id"]))]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    c1, c2 = st.columns([3, 1])
    c1.selectbox("Open a patient", [s["patient_id"] for s in sums], key="ov_open")
    c2.button("Open Digital Twin", key="btn_ov_open", on_click=lambda: select_patient(st.session_state["ov_open"]))
    scored = [s for s in sums if s["risk_score"] is not None]
    a, b = st.columns(2)
    if scored:
        fig = go.Figure(go.Bar(x=[s["patient_id"] for s in scored], y=[s["risk_score"] for s in scored], marker_color="#8b5cf6"))
        if svc.model is not None: fig.add_hline(y=svc.model.threshold, line_dash="dash", line_color="#f59e0b", annotation_text="model threshold")
        fig.update_layout(height=300, yaxis=dict(range=[0, 1.02], title="latest risk score"), margin=dict(l=10, r=10, t=30, b=10), title="Latest model risk score")
        show(fig, a)
    with b:
        st.markdown("##### Recent activity")
        ev = svc.store.get_events(limit=8)[::-1]
        theme.timeline_html([{"event_type": e["event_type"], "title": f"{e['patient_id']}: {e['title']}", "time": e["timestamp"].replace("T", " ") if e["timestamp"] else (f"hour {e['hour']}" if e["hour"] is not None else "-"),
                              "source": e["source"]} for e in ev]) if ev else st.write("No activity yet.")


def page_patients():
    theme.page_header("Patients", "Search, register and open patients"); show_flash()
    t1, t2 = st.tabs(["📋 Patient list", "➕ Register new patient"])
    with t1:
        q = st.text_input("Search by ID, name or condition", key="pat_search")
        found = svc.list_patients(q or None)
        if not found: st.info("No matching patients.")
        else:
            st.dataframe(pd.DataFrame([{"Patient ID": p["patient_id"], "Name": p["name"], "Age": p["age"], "Gender": p["gender"], "Conditions": ", ".join(p["conditions"]) or "-",
                                        "Data": "SYNTHETIC" if p["synthetic"] else ("Public record" if p["origin"] == "physionet_2019_record" else "User-entered"),
                                        "Monitoring start": (p["monitoring_start"] or "relative").replace("T", " ")} for p in found]), width="stretch", hide_index=True)
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.selectbox("Select", [p["patient_id"] for p in found], key="pat_pick", format_func=lambda i: label_of(next(p for p in found if p["patient_id"] == i)))
            c2.button("Open twin", key="btn_pat_open", on_click=lambda: select_patient(st.session_state["pat_pick"], P_TWIN))
            c3.button("Open profile", key="btn_pat_prof", on_click=lambda: select_patient(st.session_state["pat_pick"], P_PROF))
    with t2:
        theme.note("Use a demo identifier, not a real name or other personal identifiers. Only the fields below are stored, and patients are marked SYNTHETIC by default.")
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        with st.form("reg_form"):
            c1, c2, c3 = st.columns(3)
            c1.text_input("Name or demo identifier", value="Demo patient", key="reg_name", max_chars=60)
            c2.number_input("Age", 0, 120, 60, key="reg_age")
            c3.selectbox("Gender", GENDERS, index=3, key="reg_gender")
            st.multiselect("Existing medical conditions (multiple allowed)", CONDITIONS, default=["Sepsis risk monitoring"], key="reg_conditions")
            st.text_area("Relevant medical history (optional, no personal identifiers)", key="reg_history", max_chars=500)
            d1, d2, d3 = st.columns(3)
            d1.date_input("Monitoring start date", value=now.date(), key="reg_date"); d2.time_input("Monitoring start time", value=now.time(), key="reg_time", step=3600)
            d3.selectbox("ICU unit (optional)", ICU_UNITS, key="reg_unit")
            st.markdown("**Baseline health information (optional): this patient's usual values**")
            bc = st.columns(6)
            for col, k in zip(bc, ("HR", "MAP", "SBP", "Resp", "O2Sat", "Temp")):
                col.number_input(f"{SPECS[k].label} ({SPECS[k].unit})", value=None, min_value=float(SPECS[k].lo), max_value=float(SPECS[k].hi), key=f"reg_b_{k}")
            st.checkbox("This is a synthetic / demo patient (recommended)", value=True, key="reg_synth")
            st.form_submit_button("Register patient and create Digital Twin", key="btn_register", on_click=cb_register)


def page_profile():
    sn, stop = page_start("Patient Profile", "View and edit profile information")
    if stop: return
    p = sn["patient"]
    a, b = st.columns([1, 2])
    with a:
        theme.profile_card(pid[-3:], f"{p['patient_id']} · {p['name']}", f"{p['age']} years · {p['gender']}",
                           f"monitoring since {(p['monitoring_start'] or 'relative time').replace('T', ' ')}", patient_badges(p))
        st.caption(f"Registered {datetime.fromtimestamp(p['created_wall']):%Y-%m-%d %H:%M} · {len(sn['obs_records'])} observations · {len(sn['alerts'])} alerts")
    with b:
        with st.form("prof_form"):
            c1, c2, c3 = st.columns(3)
            c1.text_input("Name or demo identifier", value=p["name"], key="prof_name", max_chars=60)
            c2.number_input("Age", 0, 120, int(p["age"]), key="prof_age")
            c3.selectbox("Gender", GENDERS, index=GENDERS.index(p["gender"]), key="prof_gender")
            st.multiselect("Existing medical conditions", CONDITIONS, default=p["conditions"], key="prof_conditions")
            st.text_area("Relevant medical history", value=p["history"], key="prof_history", max_chars=500)
            c4, c5 = st.columns(2)
            c4.selectbox("ICU unit", ICU_UNITS, index=ICU_UNITS.index(p["icu_unit"]), key="prof_unit")
            c5.checkbox("Synthetic / demo patient", value=p["synthetic"], key="prof_synth")
            st.markdown("**Baseline health information (usual values)**")
            bc = st.columns(6)
            for col, k in zip(bc, ("HR", "MAP", "SBP", "Resp", "O2Sat", "Temp")):
                col.number_input(f"{SPECS[k].label} ({SPECS[k].unit})", value=p["baseline"].get(k), min_value=float(SPECS[k].lo), max_value=float(SPECS[k].hi), key=f"prof_b_{k}")
            st.form_submit_button("Save profile", key="btn_save_profile", on_click=cb_save_profile, args=(pid,))
        st.caption("Age, gender and ICU unit feed the model's demographic inputs for FUTURE observations; stored observations are not rewritten.")
    with st.expander("Patient history (events recorded for this patient)"):
        ev = svc.store.get_events(pid, ["registration", "profile", "import"])
        theme.timeline_html([{"event_type": e["event_type"], "title": e["title"], "time": (e["timestamp"] or "").replace("T", " ") or "-", "source": e["source"]} for e in ev])
    with st.expander("Danger zone"):
        st.checkbox("I understand this permanently deletes the patient and all stored data", key="del_confirm")
        st.button("Delete patient", key="btn_delete", on_click=cb_delete, args=(pid,))


# ================= pages: twin, live, prediction =================
def page_twin():
    sn, stop = page_start("Digital Twin", "Patient-specific state, baseline, history and memory")
    if stop or need_data(sn): return
    s, recs = sn["state"], sn["obs_records"]
    theme.cards([("Twin status", s["status"].replace("_", " "), "current state", "purple"), ("Observations", s["n_observations"], f"hours {s['observations_used']['first_hour']}-{s['observations_used']['last_hour']}", "info"),
                 ("Data quality", s["data_quality"], "latest observation", "ok" if s["data_quality"] == "good" else "warn"),
                 ("Latest observation", fmt_time(recs[-1]).split(" ")[-1] if recs[-1]["timestamp"] else f"hour {recs[-1]['hour']}", SOURCE_LABELS.get(recs[-1]["source"], recs[-1]["source"]), "neutral")])
    st.markdown("### Current vs previous health state")
    snaps = sn["snapshots"]; prev = snaps[-2] if len(snaps) > 1 else None
    if prev:
        st.write(f"Status: **{prev['status'].replace('_', ' ')}** → **{s['status'].replace('_', ' ')}**" +
                 (f" · risk score {prev['risk']['risk_score']:.2f} → {s['risk']['risk_score']:.2f}" if prev["risk"] and s["risk"] else ""))
    rows = []
    for v in VITALS + LABS:
        cur, old, b = s["latest_values"].get(v), (prev["latest_values"].get(v) if prev else None), s["baseline"].get(v)
        if cur is None and old is None: continue
        rows.append({"variable": f"{SPECS[v].label} ({SPECS[v].unit})", "previous": old, "current": cur, "change": (cur - old) if (cur is not None and old is not None) else None,
                     "patient baseline": b["baseline_mean"] if b else None, "last measured": "now" if s["hours_since_measured"].get(v) == 0 else f"{s['hours_since_measured'].get(v)} h ago"})
    st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
    st.markdown("### Trends: historical, current and forecast values")
    h = sn["history_df"]; avail = [v for v in VITALS + LABS if h[v].notna().any()]
    v = st.selectbox("Variable", avail, key="tw_var", format_func=lambda k: f"{SPECS[k].label} ({SPECS[k].unit})")
    show(trend_fig(sn, v))
    st.caption("Markers show OBSERVED values coloured by data source; SIMULATED/SYNTHETIC points are labelled. The ringed marker is the current value. "
               "Pink diamonds (vitals only) are statistical FORECASTS with an uncertainty band - never observations. What-if scenarios are not shown here.")
    st.markdown("### Twin memory: state at an earlier observation")
    hrs = [x["hour"] for x in snaps]
    if len(hrs) > 1:
        t = st.slider("Observation hour", min(hrs), max(hrs), max(hrs), key="mem_hour")
        old = next(x for x in reversed(snaps) if x["hour"] <= t)
        c1, c2 = st.columns(2)
        c1.write(f"**Status then:** {old['status'].replace('_', ' ')} · quality: {old['data_quality']} · observations: {old['n_observations']}")
        if old["risk"]: c1.write(f"**Risk score then:** {old['risk']['risk_score']:.2f} (used hours 0-{old['risk']['last_hour_used']})")
        c1.write("**Unusual readings then:** " + (", ".join(d["variable"] for d in old["deviations"]) or "none"))
        c2.dataframe(pd.DataFrame({"variable": VITALS, "value": [old["latest_values"].get(x) for x in VITALS]}).round(2), hide_index=True, width="stretch")
    st.markdown("### Observation sources and timestamps")
    rk = {r["hour"]: r["risk_score"] for r in sn["risk_history"]}
    st.dataframe(pd.DataFrame([{"time": fmt_time(r), "hour": r["hour"], "source": SOURCE_LABELS.get(r["source"], r["source"]),
                                "simulated / synthetic": "YES" if r["source"] in SYNTHETIC_SOURCES else "no", "model risk": rk.get(r["hour"])} for r in recs[-12:]][::-1]).round(2),
                 width="stretch", hide_index=True)


def page_live():
    sn, stop = page_start("Live Monitoring", "Enter observations or run a simulated feed; the twin updates after each one")
    if stop: return
    theme.note("No hospital system or wearable is connected. Manual entries are exactly what you type; simulated feeds are generated data and are labelled SIMULATED LIVE DATA.")
    t1, t2 = st.tabs(["✍️ Manual observation entry", "🎛 Simulated live monitoring"])
    with t1:
        n = st.session_state.get("obs_n", 0); sugg = svc._next_timestamp(pid)
        with st.form("obs_form"):
            c1, c2 = st.columns(2)
            c1.date_input("Observation date", value=sugg.date(), key=f"obs_date_{n}")
            c2.time_input("Observation time", value=sugg.time().replace(second=0), key=f"obs_time_{n}", step=60)
            st.markdown("**Vital signs**"); cols = st.columns(4)
            for i, k in enumerate(VITALS):
                sp = SPECS[k]; cols[i % 4].number_input(f"{sp.label} ({sp.unit})", value=None, min_value=float(sp.lo), max_value=float(sp.hi), step=0.1, format="%.1f", key=f"obs_{k}_{n}")
            with st.expander("Laboratory values (optional, enter only what was measured)"):
                cols = st.columns(3)
                for i, k in enumerate(LABS):
                    sp = SPECS[k]; cols[i % 3].number_input(f"{sp.label} ({sp.unit})" if sp.unit else sp.label, value=None, min_value=float(sp.lo), max_value=float(sp.hi),
                                                          step=0.01 if sp.hi <= 10 else (0.1 if sp.hi <= 300 else 1.0), format="%.2f" if sp.hi <= 10 else "%.1f", key=f"obs_{k}_{n}")
            st.caption("Blank = not measured. Blanks are stored as missing and handled by the model's existing preprocessing (carry-forward, training median, missing indicators); "
                       "nothing is filled in silently. Age, gender and ICU stay hour come from the profile.")
            st.form_submit_button("Save observation", key="btn_save_obs", on_click=cb_save_obs, args=(pid,))
        lo = st.session_state.get("_last_obs")
        if lo and lo["pid"] == pid:
            r = lo["risk"]
            st.success(f"Saved at hour {lo['hour']} ({(lo['ts'] or '').replace('T', ' ')}). Twin status: {lo['status'].replace('_', ' ')}." +
                       (f" Sepsis model score {r['risk_score']:.2f} ({r['risk_category']})." if r else " No model loaded, so no prediction."))
            if lo["note"]: st.info(lo["note"])
            for sev, trig in lo["alerts"]: st.warning(f"New {sev.upper()} alert: {trig}")
    with t2:
        fs = svc.feed_status(pid)
        st.markdown(theme.badge("SIMULATED LIVE DATA", "warn") + " Generated observations (scenario drifts + noise) are added to this patient's twin one by one.", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k], key="feed_scenario")
        c2.slider("Simulated hours to generate", 3, 48, 12, key="feed_hours")
        c3.slider("Seconds per simulated hour", 0.2, 5.0, 1.0, 0.2, key="feed_interval")
        b1, b2, b3, b4 = st.columns(4)
        b1.button("▶ Start simulated feed", key="btn_feed_start", on_click=cb_start_feed, args=(pid,))
        if b2.button("⏸ Pause", key="btn_feed_pause"): svc.pause_feed(pid)
        if b3.button("⏯ Resume", key="btn_feed_resume"): svc.resume_feed(pid)
        if b4.button("⏹ Stop", key="btn_feed_stop"): svc.stop_feed(pid)
        if svc.replay_queues.get(pid):
            st.button(f"▶ Replay the remaining {len(svc.replay_queues[pid])} hours of the public record (simulation)", key="btn_hist_replay",
                      on_click=lambda: svc.start_historical_replay(pid, float(st.session_state.get("feed_interval", 1.0))))
        if fs:
            st.progress(fs["position"] / fs["total"] if fs["total"] else 0.0, text=f"{fs['label']}: {fs['status']} ({fs['position']}/{fs['total']})")
            if fs["error"]: st.error(fs["error"])
    sn = svc.snapshot(pid)
    if sn["has_data"]:
        st.markdown("### Latest observations")
        h, rk = sn["history_df"].set_index("hour"), {r["hour"]: r["risk_score"] for r in sn["risk_history"]}
        rows = []
        for r in sn["obs_records"][-15:][::-1]:
            row = h.loc[r["hour"]]
            rows.append({"time": fmt_time(r), "source": SOURCE_LABELS.get(r["source"], r["source"]), "simulated": "YES" if r["source"] in SYNTHETIC_SOURCES else "no",
                         **{k: row[k] for k in ("HR", "Resp", "MAP", "O2Sat", "Temp", "SBP")}, "model risk": rk.get(r["hour"])})
        st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
        chosen = st.multiselect("Chart", VITALS, default=["HR", "MAP", "O2Sat"], key="live_chart_vars")
        for v in chosen: show(trend_fig(sn, v, horizon_fc=False))
    else:
        st.info("No observations yet. Save the first one above.")


@st.cache_data(show_spinner=False)
def cached_ecg_record(records_dir, record):
    from src.ecg.loader import load_record
    return load_record(records_dir, record)


def ecg_report_table():
    p = resolve_path(cfg, "reports_dir") / "ecg_metrics.json"
    if p.exists():
        rep = json.loads(p.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"model": n, "macro-F1": r["test"]["macro_f1_present_classes"], "balanced acc": r["test"]["balanced_accuracy"]} for n, r in rep["results"].items()]).round(3), width="stretch", hide_index=True)
    else: st.info("No ECG evaluation report yet. Run `python scripts/train_ecg.py`.")



def ecg_panel():
    from src.ecg.loader import list_records
    from src.ecg.train import DS1, DS2
    st.caption("A separate record-level module: MIT-BIH recordings are different people from registered patients and are not linked to any patient twin.")
    rdir = resolve_path(cfg, "raw_dir") / cfg["datasets"]["ecg"]["raw_subdir"]; recs = list_records(rdir)
    if not recs:
        st.info("ECG recordings are not included in this deployment, so the signal viewer is off. The trained model's evaluation is shown below. To view recordings locally run `python scripts/download_ecg.py`.")
        ecg_report_table(); return
    rec_id = st.selectbox("Record", recs, key="ecg_rec")
    st.write(f"**ecg:{rec_id}** - " + ("held-out TEST record" if rec_id in DS2 else "training/validation record (not an honest test)" if rec_id in DS1 else "outside the DS1/DS2 split"))
    rec = cached_ecg_record(str(rdir), rec_id); em = assets.get("ecg_model"); res = em.classify_record(rec) if em else None
    if em is None: st.warning(f"ECG model not loaded ({assets.get('ecg_error')}). Showing signal and annotations only.")
    fs_ = rec["fs"]; dur = len(rec["signal"]) / fs_
    t0 = st.slider("Start (seconds)", 0.0, max(1.0, dur - 10), 0.0, 5.0, key="ecg_t0"); i0, i1 = int(t0 * fs_), int((t0 + 10) * fs_)
    fig = go.Figure(go.Scatter(x=[x / fs_ for x in range(i0, min(i1, len(rec["signal"])))], y=rec["signal"][i0:i1], mode="lines", name=f"lead {rec['lead']}", line=dict(color="#6366f1")))
    seg = [(a, b) for a, b in zip(rec["samples"], rec["symbols"]) if i0 <= a < i1]
    fig.add_trace(go.Scatter(x=[a / fs_ for a, _ in seg], y=[rec["signal"][a] for a, _ in seg], mode="markers+text", text=[b for _, b in seg], textposition="top center", marker=dict(size=7, color="#94a3b8"), name="annotation"))
    if res is not None and len(res["samples"]):
        m = (res["samples"] >= i0) & (res["samples"] < i1); bad = m & (res["pred"] != res["true"])
        fig.add_trace(go.Scatter(x=res["samples"][bad] / fs_, y=rec["signal"][res["samples"][bad]], mode="markers", marker=dict(size=14, color="#ef4444", symbol="x"), name="model disagrees"))
    fig.update_layout(height=340, xaxis_title="seconds"); show(fig)
    ecg_report_table()


def eval_panel():
    p = resolve_path(cfg, "reports_dir") / "sepsis_metrics.json"
    if not p.exists(): st.warning("No evaluation report found. Run `python scripts/train_sepsis.py`."); return
    rep = json.loads(p.read_text(encoding="utf-8")); m = rep["meta"]
    st.write(f"**Selected model:** {m['selected_model']} ({m['selection_rule']}) · seed {m['seed']} · patients per split: {m['split_patients']}")
    st.dataframe(pd.DataFrame([{"model": n, "AUROC": r["test"]["auroc"], "AUPRC": r["test"]["auprc"], "random AUPRC": r["test"]["auprc_random_baseline"], "precision": r["test"]["precision"],
                                "recall": r["test"]["recall"], "F1": r["test"]["f1"]} for n, r in rep["results"].items()]).round(3), width="stretch", hide_index=True)
    st.caption("Held-out test patients from the PhysioNet 2019 data. Modest discrimination, low precision and recall; scores are ranking scores, not calibrated probabilities; no external validation.")


def tabular_panel(key, sn):
    """Patient-linked point-in-time estimate using the disease's own tabular model."""
    reg = svc.registry[key]; tm = (assets.get("tabular") or {}).get(key)
    if reg.status == "unavailable" or tm is None:
        st.markdown(theme.badge("MODEL UNAVAILABLE", "neutral"), unsafe_allow_html=True)
        st.info(f"**{reg.name}: no model available.** {reg.reason}"); st.caption(f"Dataset: {reg.dataset}. The sepsis model is never applied to this condition."); return
    if reg.status == "pipeline_test":
        st.error("PIPELINE TEST MODEL - NOT VALIDATED. " + reg.reason)
    else:
        st.markdown(theme.badge("MODEL AVAILABLE (trained and evaluated on public data)", "ok"), unsafe_allow_html=True)
    mi = reg.model_info
    st.caption(f"Task: {reg.description}. Dataset: {reg.dataset}. Model: {mi.get('name')} · threshold {mi.get('threshold'):.3f} · data origin: {mi.get('data_origin')} · trained {mi.get('trained_at')}.")
    st.caption("Enter this assessment's clinical values. Missing fields use training-set medians and are disclosed. With a selected patient and a validated model, the assessment is stored in that patient's Digital Twin history. This is a cross-sectional estimate, not a temporal forecast.")
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(tm.spec.features):
        with cols[i % 3]:
            if f.kind == "bool":
                v = st.selectbox(f.label, ["not given", "No", "Yes"], key=f"tab_{key}_{f.name}"); vals[f.name] = None if v == "not given" else (1 if v == "Yes" else 0)
            else:
                vals[f.name] = st.text_input(f"{f.label}" + (f" ({f.unit})" if f.unit else ""), key=f"tab_{key}_{f.name}", placeholder=f"{f.lo:g} - {f.hi:g}")
    if st.button("Assess selected patient" if pid else "Estimate risk (stand-alone)", key=f"tab_go_{key}"):
        try:
            if pid and reg.status == "available":
                r = svc.assess_disease(pid, key, vals)
                st.success("Assessment saved to the selected patient's Digital Twin history.")
            else:
                r = tm.predict(vals)
                if pid:
                    st.warning("This model is not validated, so this pipeline-test result is not saved to the patient's clinical history.")
            theme.cards([("Risk score", f"{r['risk_score']:.2f}", f"threshold {r['threshold']:.2f}", "crit" if r["flagged"] else "ok"),
                         ("Category", r.get("category", r.get("risk_category", "—")), "relative to the model threshold", "neutral"),
                         ("Inputs given", f"{r['n_inputs_given']}/{r['n_inputs_total']}", "missing -> median", "neutral")])
            if r.get("imputation_note"): st.caption(r["imputation_note"] + " Missing: " + ", ".join(r["missing_inputs"]))
            if r.get("population_warning"): st.warning(r["population_warning"])
            st.caption(r["calibration"] + ". " + r["disclaimer"])
            if reg.status == "available":
                with st.expander("Explain this estimate (SHAP)", expanded=True):
                    try:
                        from src.diseases.explain import explain_tabular_prediction
                        sx = explain_tabular_prediction(tm, vals)
                        st.write(f"Baseline score: **{sx['base_value']:.3f}** · Current model score: **{sx['predicted_score']:.3f}**")
                        st.caption(sx["method"] + f" · evaluation budget {sx['max_evals']}")
                        show_items = sx["items"][:10]
                        if show_items:
                            sdf = pd.DataFrame([{
                                "Feature": x["label"], "Input": "Missing (training median used)" if x["missing_input"] else x["input_value"],
                                "SHAP contribution": x["shap_value"], "Direction": x["direction"]
                            } for x in show_items])
                            st.dataframe(sdf.round(4), width="stretch", hide_index=True)
                        st.info(sx["note"])
                    except Exception as ex:
                        st.warning(f"SHAP explanation unavailable for this run: {type(ex).__name__}: {ex}")
        except (ValueError, RuntimeError, KeyError) as e:
            st.error(str(e))
    if pid:
        history = svc.disease_risk_history(pid, key).get(key, [])
        if history:
            st.markdown("**This patient's assessment history**")
            st.dataframe(pd.DataFrame([{"Assessment": x["assessment_id"], "Twin hour": x["hour"],
                                        "Risk": x.get("risk_score"), "Category": x.get("category"),
                                        "Inputs supplied": x.get("n_inputs_given")} for x in history]).round(3),
                         width="stretch", hide_index=True)
    with st.expander("Model evaluation (held-out test split) and limitations"):
        rp = resolve_path(cfg, "reports_dir") / f"{key}_metrics.json"
        if rp.exists():
            rep = json.loads(rp.read_text(encoding="utf-8")); t_ = rep["results"]["test"]; cm = t_["confusion_matrix"]
            st.write(f"AUROC {t_['auroc']:.3f} (95% CI {t_['auroc_ci95'][0]:.3f}-{t_['auroc_ci95'][1]:.3f}) · AUPRC {t_['auprc']:.3f} (random {t_['auprc_random_baseline']:.3f}) · precision {t_['precision']:.3f} · recall {t_['recall']:.3f} · F1 {t_['f1']:.3f} · test n={t_['n_rows']}")
            st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], index=["actual 0", "actual 1"], columns=["predicted 0", "predicted 1"]), width="stretch")
            for w in rep.get("warnings", []): st.warning(w)
        for l in mi.get("limitations", []): st.write("- " + l)


def page_prediction():
    theme.page_header("Disease Prediction", "Each disease keeps its own model, inputs and evaluation"); show_flash()
    sn = svc.snapshot(pid) if pid else None
    tabs = st.tabs(["🦠 Sepsis", "🩸 Type 2 Diabetes", "🫀 Cardiovascular", "🩺 Hypertension", "💓 ECG (record-level)"])
    with tabs[0]:
        reg = svc.registry["sepsis"]
        if not pid: need_patient()
        elif reg.status == "unavailable": st.warning(reg.reason)
        elif not need_data(sn):
            if reg.status == "pipeline_test":
                st.error("PIPELINE TEST MODEL - NOT VALIDATED. " + reg.reason + " Scores below are NOT real-data performance and must not be presented as results.")
            s, r, p = sn["state"], sn["state"]["risk"], sn["patient"]
            theme.cards([("Patient", pid, p["name"], "neutral"), ("Disease", "Sepsis", "ICU, hourly model", "purple"),
                         ("Risk score", f"{r['risk_score']:.2f}", f"threshold {r['threshold']:.2f}", tone(sn)), ("Risk category", r["risk_category"], "relative to model threshold", tone(sn))])
            c1, c2 = st.columns(2)
            c1.markdown(f"**Prediction time:** {(r.get('data_timestamp') or 'hour ' + str(r['last_hour_used'])).replace('T', ' ')} (data time) · computed {r.get('predicted_at', '-').replace('T', ' ')}")
            c1.markdown(f"**Inputs:** observations up to hour {r['last_hour_used']} ({r['n_observations_used']} observations) - no later data; SepsisLabel is never an input.")
            mi = reg.model_info
            c1.markdown(f"**Training data:** {mi.get('data_origin') or '-'} · trained {mi.get('trained_at') or '-'} · " + ("evaluation report verified" if mi.get("validated") else "NOT validated"))
            c2.markdown(f"**Model:** {mi['name']} · threshold {mi['threshold']:.3f} (set on validation data)" if mi.get("threshold") is not None else f"**Model:** {mi['name']}")
            c2.markdown(f"**Selection:** {mi.get('selected_by') or '-'} · patients per split {mi.get('patients_per_split') or '-'}")
            show(risk_fig(sn))
            st.caption("The score is a model ranking score, not a probability. Categories (Low / Elevated / High / Critical) are display bands relative to the model threshold and a configurable critical level.")
            st.markdown("#### Why this score? (occlusion analysis)")
            x = svc.explain(pid)
            if x and x["items"]:
                df = pd.DataFrame(x["items"]); df["variable"] = df["feature"].map(lambda k: SPECS[k].label)
                fig = go.Figure(go.Bar(x=df["effect"][::-1], y=df["variable"][::-1], orientation="h", marker_color=["#ef4444" if e > 0 else "#3b82f6" for e in df["effect"][::-1]]))
                fig.update_layout(height=280, xaxis_title="change in score if this variable were at its typical (median) value → positive = pushes the score up", margin=dict(l=10, r=10, t=10, b=10))
                show(fig); st.caption(x["note"] + f" Computed with the real model ({x['model_name']}) at hour {x['hour']}.")
            else: st.info("No explanation available yet.")
            if r.get("top_factors"):
                st.write("Linear model coefficient × value: " + ", ".join(f"{f['feature']} ({f['contribution']:+.2f})" for f in r["top_factors"]))
            with st.expander("Model evaluation (held-out test patients)"): eval_panel()
    for tab, key in zip(tabs[1:4], ("diabetes", "cardiovascular", "hypertension")):
        with tab: tabular_panel(key, sn)
    with tabs[4]: ecg_panel()


# ================= pages: timeline, what-if, insights, alerts, data =================
def page_timeline():
    sn, stop = page_start("Patient Timeline", "Chronological history of everything that happened to this patient")
    if stop: return
    ev = svc.store.get_events(pid)
    types = sorted({e["event_type"] for e in ev})
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Event types", types, default=types, key="tl_types", format_func=lambda t: t.replace("_", " "))
    order = c2.radio("Order", ["Oldest first", "Newest first"], key="tl_order", horizontal=True)
    hrs = [e["hour"] for e in ev if e["hour"] is not None]
    lo, hi = (min(hrs), max(hrs)) if hrs else (0, 1)
    rng = st.slider("Observation hours", lo, max(hi, lo + 1), (lo, max(hi, lo + 1)), key="tl_range") if hrs else (lo, hi)
    shown = [e for e in ev if e["event_type"] in chosen and (e["hour"] is None or rng[0] <= e["hour"] <= rng[1])]
    if order == "Newest first": shown = shown[::-1]
    st.caption(f"{len(shown)} of {len(ev)} events. Simulated what-if scenarios are marked as such and are never observations.")
    items = [{"event_type": e["event_type"], "title": e["title"], "source": SOURCE_LABELS.get(e["source"], e["source"] or ""),
              "time": (e["timestamp"].replace("T", " ") if e["timestamp"] else (f"hour {e['hour']} (relative)" if e["hour"] is not None else datetime.fromtimestamp(e["wall_time"]).strftime("%Y-%m-%d %H:%M")))} for e in shown]
    theme.timeline_html(items) if items else st.info("No events match the filters.")
    if shown:
        st.download_button("Download timeline (CSV)", pd.DataFrame([{k: e[k] for k in ("event_id", "hour", "timestamp", "event_type", "title", "severity", "source")} for e in shown]).to_csv(index=False),
                           file_name=f"{pid}_timeline.csv", key="dl_timeline")


def page_whatif():
    sn, stop = page_start("What-if Simulation", "Compare the patient's current state with a hypothetical one")
    if stop or need_data(sn): return
    st.markdown(theme.badge("SIMULATED SCENARIO", "pink") + " Scenarios are hypothetical and are stored separately from real observations.", unsafe_allow_html=True)
    theme.note("The scenario values are scored by the same trained model. This shows how the model responds; it is not a treatment effect, a recommendation or a medical prediction.")
    t1, t2, t3 = st.tabs(["🎚 Current-state scenario", "📈 Trajectory scenario", "🗂 Saved scenarios"])
    s = sn["state"]; hr = s["hour"]
    with t1:
        st.caption(f"Current state = the latest known values at hour {hr}. Change a vital to create a hypothetical state, e.g. MAP 65 → 75.")
        cols, changes = st.columns(4), {}
        for i, v in enumerate(["HR", "Resp", "Temp", "O2Sat", "SBP", "MAP", "DBP", "EtCO2"]):
            sp, cur = SPECS[v], s["latest_values"].get(v)
            val = cols[i % 4].number_input(f"{sp.label} ({sp.unit}) - current " + (f"{cur:.1f}" if cur is not None else "not measured"), min_value=float(sp.lo), max_value=float(sp.hi),
                                          value=round(cur, 1) if cur is not None else None, step=0.1, format="%.1f", key=f"wi_{v}_{hr}")
            if val is not None and (cur is None or abs(val - cur) > 1e-9): changes[v] = val
        k = st.slider("Apply to the last N observations", 1, min(6, s["n_observations"]), 1, key="wi_k") if s["n_observations"] > 1 else 1
        if st.button("Run scenario", key="btn_whatif"):
            try: st.session_state["wi_res"] = (pid, hr, svc.whatif(pid, changes, k))
            except Exception as e: st.error(f"Could not run scenario: {e}")
        res = st.session_state.get("wi_res")
        if res and res[0] == pid and res[1] == hr:
            r = res[2]
            if r.get("reason"): st.warning(r["reason"])
            c, sc = r["current"], r["scenario"]
            theme.cards([("Current model score", f"{c['risk_score']:.2f}" if c else "n/a", svc.risk_category(c["risk_score"], c["threshold"]) if c else "no model", "info"),
                         ("Simulated model score", f"{sc['risk_score']:.2f}" if sc else "n/a", svc.risk_category(sc["risk_score"], sc["threshold"]) if sc else "not calculated", "pink"),
                         ("Difference", f"{r['delta_score']:+.2f}" if r["delta_score"] is not None else "n/a", "simulated minus current", "purple")])
            tb = r["table"]; fig = go.Figure()
            fig.add_trace(go.Bar(x=tb["variable"], y=tb["current"], name="Current (observed state)", marker_color="#6366f1"))
            fig.add_trace(go.Bar(x=tb["variable"], y=tb["simulated"], name="Simulated (hypothetical)", marker_color="#ec4899"))
            fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10)); show(fig)
            st.dataframe(tb[tb["current"].notna() | tb["simulated"].notna()].round(2), width="stretch", hide_index=True)
            st.caption(r["note"])
    with t2:
        st.caption("Shift selected vitals from an earlier hour onward and re-score every observed hour (each using only data up to that hour).")
        start = st.slider("Apply changes from hour", 0, hr, max(0, hr - 6), key="wi_start") if hr > 0 else 0
        cols, adj = st.columns(3), {}
        for i, (v, (nm, unit, rng)) in enumerate(ADJUSTABLE.items()):
            adj[v] = cols[i % 3].slider(f"{nm} change ({unit})", -rng, rng, 0.0, rng / 20, key=f"wi_d_{v}")
        if st.button("Run trajectory scenario", key="btn_traj"):
            try: st.session_state["wi_traj"] = (pid, hr, svc.whatif_trajectory(pid, adj, start))
            except Exception as e: st.error(str(e))
        tr = st.session_state.get("wi_traj")
        if tr and tr[0] == pid and tr[1] == hr:
            cmp = tr[2]; fig = go.Figure()
            fig.add_trace(go.Scatter(x=cmp["baseline"]["hour"], y=cmp["baseline"]["risk_score"], name="observed", line=dict(color="#6366f1", width=3)))
            fig.add_trace(go.Scatter(x=cmp["scenario"]["hour"], y=cmp["scenario"]["risk_score"], name="scenario (SIMULATED)", line=dict(color="#ec4899", width=3, dash="dash")))
            fig.add_hline(y=cmp["threshold"], line_dash="dot", line_color="#f59e0b", annotation_text="model threshold")
            fig.update_layout(height=340, yaxis=dict(range=[0, 1.02], title="model risk score"), xaxis_title="observation hour"); show(fig)
            for line in explain_whatif(cmp): st.write("- " + line)
    with t3:
        runs = svc.store.get_whatifs(pid)
        if not runs: st.info("No saved scenarios for this patient yet.")
        else:
            st.dataframe(pd.DataFrame([{"run": r["run_id"], "time": datetime.fromtimestamp(r["wall_time"]).strftime("%Y-%m-%d %H:%M:%S"), "hour": r["hour"],
                                        "changes": ", ".join(f"{k} → {v:g}" for k, v in r["changes"].items()) or "-", "current score": r["current_score"],
                                        "simulated score": r["scenario_score"], "type": "SIMULATED"} for r in runs][::-1]).round(3), width="stretch", hide_index=True)


def page_insights():
    sn, stop = page_start("AI Insights", "Evidence-based observations about this patient")
    if stop: return
    st.caption("Generated locally from this patient's own data (profile, observations, baseline, model score, occlusion analysis, anomaly detector, alerts). "
               "Each insight names its source and time. No LLM and no API key are used; nothing here is a diagnosis.")
    use = st.checkbox("Include model occlusion analysis (a few seconds)", value=True, key="ins_occ")
    items = svc.insights(pid, with_occlusion=use)
    for sev, title in (("notable", "Notable"), ("attention", "Worth attention"), ("info", "Context")):
        grp = [i for i in items if i["severity"] == sev]
        if grp: st.markdown(f"### {title}"); theme.insight_cards(grp)


def agent_panel():
    """Agentic triage: ranks patients from evidence, drafts PROPOSED follow-ups. Nothing is executed; each proposal needs a human decision."""
    over = Oversight(svc)
    with st.expander("🤖 Triage agent: priority ranking and proposals (human approval required)"):
        st.caption("Deterministic and offline: scores come from model outputs, alerts and data freshness (weights in config). It only PROPOSES; every proposal waits for your approve/reject decision, "
                   "and approving records the decision - it triggers no clinical action. Suggestions are not diagnoses.")
        if st.button("Run triage now", key="agent_run"):
            r = over.run(by="dashboard_user"); st.session_state["agent_rank"] = r["ranking"]; st.session_state["agent_new"] = r["new_proposals"]
        rank = st.session_state.get("agent_rank")
        if rank:
            st.write(f"{st.session_state.get('agent_new', 0)} new proposal(s) created.")
            st.dataframe(pd.DataFrame([{"Patient": r["patient_id"], "Name": r["name"], "Priority": round(r["priority"]), "Tier": r["tier"], "Why": "; ".join(r["reasons"][:3]) or "no threshold exceeded"} for r in rank]),
                         width="stretch", hide_index=True)
            pick = st.selectbox("Evidence for", [r["patient_id"] for r in rank], key="agent_ev")
            e = next(r for r in rank if r["patient_id"] == pick)
            st.write(e["narrative"])
            st.dataframe(pd.DataFrame(e["evidence"]), width="stretch", hide_index=True) if e["evidence"] else st.caption("No evidence items.")
        props = over.proposals(None, "proposed")
        st.markdown(f"**Open proposals ({len(props)})**")
        for q in props:
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.write(f"**{q['patient_id']} - {q['title']}**  \n{q['rationale']}")
            if c2.button("Approve", key=f"pa_{q['proposal_id']}"): over.decide(q["proposal_id"], "approved", "dashboard_user"); st.rerun()
            if c3.button("Reject", key=f"pr_{q['proposal_id']}"): over.decide(q["proposal_id"], "rejected", "dashboard_user"); st.rerun()


def page_alerts():
    theme.page_header("Smart Alerts", "Automatic alerts with human review"); show_flash()
    with st.expander("How alerts are generated"):
        st.write(PROTOTYPE_RULES)
        a = svc.alerts
        st.write(f"Active rules (all configurable): model score ≥ model threshold → warning/critical (critical at {a.critical_threshold:.2f}); score rising by ≥ {a.risk_rise_delta:.2f} within "
                 f"{a.risk_rise_window} observations → warning; ≥ {a.multi_vital_min} vitals unusual versus the patient's own baseline at once → warning; one unusual vital → info. "
                 f"Duplicates are merged for {a.cooldown_hours} h; unacknowledged critical alerts escalate after {a.escalate_after_hours} h. Vital-sign alerts are not labelled as sepsis alerts.")
    agent_panel()
    scope = st.radio("Show alerts for", ["Selected patient", "All patients"], horizontal=True, key="alert_scope")
    al = svc.all_alerts() if (scope == "All patients" or not pid) else svc.snapshot(pid)["alerts"]
    al = sorted(al, key=lambda a: (a["status"] == "resolved", SEV_RANK[a["severity"]], -a["hour"]))
    theme.cards([("Needs review", sum(a["status"] in ("active", "escalated") for a in al), "active or escalated", "crit"), ("Critical", sum(a["severity"] == "critical" for a in al), "", "crit"),
                 ("Warning", sum(a["severity"] == "warning" for a in al), "", "warn"), ("Info", sum(a["severity"] == "info" for a in al), "", "info")])
    if not al: st.info("No alerts yet. They appear automatically when new observations meet an alert rule."); return
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning", "info"], key="sev_f")
    stat = f2.multiselect("Status", ["active", "acknowledged", "escalated", "resolved"], default=["active", "acknowledged", "escalated", "resolved"], key="stat_f")
    shown = [a for a in al if a["severity"] in sev and a["status"] in stat]
    st.dataframe(pd.DataFrame([{"Alert": a["alert_id"], "Patient": a["patient_id"], "Time": (a.get("data_timestamp") or f"hour {a['hour']}").replace("T", " "), "Severity": a["severity"],
                                "Type": a["type"], "Domain": a["domain"], "Status": a["status"], "Score": a["score"], "Triggering condition": a["trigger"]} for a in shown]).round(2),
                 width="stretch", hide_index=True)
    openl = [a["alert_id"] for a in al if a["status"] != "resolved"]
    if openl:
        st.markdown("### Review an alert")
        pick = st.selectbox("Alert", openl, key="alert_pick"); a = next(x for x in al if x["alert_id"] == pick)
        d1, d2 = st.columns(2)
        d1.markdown(f"**{a['severity'].upper()}** · {a['type']} · patient {a['patient_id']} · status **{a['status']}**"); d1.write(a["explanation"])
        d2.markdown("**Relevant observations**" + (f" ({(a.get('data_timestamp') or 'hour ' + str(a['hour'])).replace('T', ' ')}, {SOURCE_LABELS.get(a.get('data_source'), a.get('data_source') or '-')})"))
        d2.dataframe(pd.DataFrame([{"variable": SPECS[k].label if k in SPECS else k, "value": v} for k, v in (a.get("relevant_observations") or {}).items()]), hide_index=True, width="stretch")
        st.radio("Action", ["Acknowledge", "Escalate", "Resolve"], horizontal=True, key="alert_action")
        st.text_input("Note (optional)", key="alert_note")
        st.button("Apply", key="btn_ack", on_click=cb_alert_action)
        with st.expander("Audit trail"):
            st.dataframe(pd.DataFrame(a["history"]).assign(time=lambda d: pd.to_datetime(d["wall_time"], unit="s").dt.strftime("%H:%M:%S"))[["time", "status", "by", "note"]], width="stretch", hide_index=True)
    st.caption("This is decision support: acknowledging or escalating only changes the review status. No notification is sent and no treatment is triggered.")


def page_data():
    theme.page_header("Data Management", "Import, upload, export and system status"); show_flash()
    t1, t2, t3, t4, t5 = st.tabs(["📥 Import public record", "🧪 Demo patients", "📤 Upload history", "💾 Export", "⚙️ System status"])
    with t1:
        if svc.df is None: st.info(f"The public dataset is not prepared ({assets.get('data_error')}). Run `python scripts/prepare_sepsis_data.py`.")
        else:
            st.caption("Register a real, de-identified ICU record from the held-out test patients as a patient. The first hours are loaded now; the rest can be replayed on the Live page (simulation).")
            st.selectbox("Record", svc.splits["test"][:300], key="imp_record"); st.slider("Hours to load now", 1, 40, 12, key="imp_n")
            st.button("Import as patient", key="btn_import", on_click=cb_import_hist)
    with t2:
        st.write("Creates three SYNTHETIC patients (stable, deteriorating, recovering) with 24 hours of generated history. They are labelled synthetic everywhere.")
        st.button("Create 3 synthetic demo patients", key="btn_demo", on_click=cb_demo)
    with t3:
        if not pid: need_patient("_up")
        else:
            theme.note(f"Upload hourly readings for {pid}. Hours must continue after the patient's last observation. Uploaded rows are labelled 'Uploaded file'.")
            st.download_button("Download blank template (CSV)", template_csv(), file_name="healthtwin_template.csv", key="dl_tpl")
            up = st.file_uploader("Patient file (.csv, .psv, .txt)", type=["csv", "psv", "txt"], key="uploader")
            if up is not None:
                try: raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=None, engine="python")
                except Exception as e: st.error(f"Could not read the file: {e}"); raw = None
                if raw is not None:
                    st.dataframe(raw.head(15), width="stretch")
                    clean, errors, warns = validate_patient_upload(raw, Path(up.name).stem)
                    for e in errors: st.error(e)
                    for w in warns: st.warning(w)
                    if clean is not None:
                        st.session_state["upload_clean"] = clean; st.success(f"Valid: {len(clean)} hourly rows.")
                        st.button("Add to patient", key="btn_upload_add", on_click=cb_upload_patient, args=(pid,))
    with t4:
        if not pid: need_patient("_ex")
        else:
            sn = svc.snapshot(pid); h = sn["history_df"]
            if sn["has_data"]:
                meta = {r["hour"]: r for r in sn["obs_records"]}
                obs = h.drop(columns=["subject_id"]).copy(); obs.insert(1, "timestamp", obs["hour"].map(lambda x: meta[x]["timestamp"])); obs.insert(2, "source", obs["hour"].map(lambda x: meta[x]["source"]))
                obs.insert(3, "simulated_or_synthetic", obs["source"].isin(SYNTHETIC_SOURCES))
                st.download_button("Observations (CSV)", obs.to_csv(index=False), file_name=f"{pid}_observations.csv", key="dl_obs")
            st.download_button("Timeline events (CSV)", pd.DataFrame(svc.store.get_events(pid)).drop(columns=["detail"], errors="ignore").to_csv(index=False), file_name=f"{pid}_events.csv", key="dl_ev")
            st.download_button("Alerts (CSV)", pd.DataFrame(sn["alerts"]).drop(columns=["history", "factors", "relevant_observations"], errors="ignore").to_csv(index=False), file_name=f"{pid}_alerts.csv", key="dl_al")
    with t5:
        cnt = svc.store.counts(); reg = svc.registry
        st.dataframe(pd.DataFrame([{"component": "Sepsis model", "status": reg["sepsis"].status, "detail": reg["sepsis"].reason or str(reg["sepsis"].model_info.get("name"))},
                                   {"component": "Forecast models", "status": ("available" if assets.get("gates", {}).get("forecast", {}).get("passed") else "pipeline_test") if assets.get("forecaster") is not None else "unavailable", "detail": assets.get("forecast_error") or "6 vitals, +1 h / +3 h"},
                                   {"component": "ECG model", "status": reg["ecg"].status, "detail": reg["ecg"].reason or "record-level module"},
                                   {"component": "Public dataset (processed)", "status": ("available" if (assets.get("data_provenance") or {}).get("data_origin") in ("real_public", "user_confirmed_real") else "unverified origin") if svc.df is not None else "unavailable", "detail": assets.get("data_error") or f"{svc.df.subject_id.nunique()} patients · origin: {(assets.get('data_provenance') or {}).get('data_origin')}"},
                                   {"component": "Patient database (SQLite)", "status": "available", "detail": f"{cnt['observations']} observations · {cnt['risks']} predictions · {cnt['alerts']} alerts"}]),
                     width="stretch", hide_index=True)


PAGES = {P_OV: page_overview, P_PATS: page_patients, P_PROF: page_profile, P_TWIN: page_twin, P_LIVE: page_live, P_PRED: page_prediction, P_TIME: page_timeline,
         P_WHAT: page_whatif, P_INS: page_insights, P_ALERT: page_alerts, P_DATA: page_data}
PAGES[page]()
if page == P_DATA:
    theme.footer(DISCLAIMER)     # one short note on the Data page instead of every page
if svc.any_feed_running():            # keep the whole page (sidebar, navigation, data) in sync while a simulated feed runs
    time.sleep(1.2)
    st.rerun()
