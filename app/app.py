import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import pickle
import time
import sys, os

sys.path.append(os.getcwd())
from agent import HealthTwinAgent, AlertLevel
from communication import CommunicationModule

DATA_DIR = "/content/drive/MyDrive/healthtwin_sepsis_data"
st.set_page_config(page_title="HealthTwin", layout="wide")
st.title("HealthTwin — Remote Patient Monitoring")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SmallSepsisTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        padding_mask = (mask == 0)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.classifier(out).squeeze(-1)


@st.cache_resource
def load_everything():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_seq = np.load(f"{DATA_DIR}/test_seq.npy")
    test_mask = np.load(f"{DATA_DIR}/test_mask.npy")
    test_labels = np.load(f"{DATA_DIR}/test_labels.npy")
    test_df = pd.read_csv(f"{DATA_DIR}/test_preprocessed.csv")
    test_pids = test_df['PatientID'].unique().tolist()
    n_features = test_seq.shape[2]
    model = SmallSepsisTransformer(n_features=n_features).to(device)
    model.load_state_dict(torch.load(f"{DATA_DIR}/sepsis_transformer_small_best.pth", map_location=device))
    model.eval()

    with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{DATA_DIR}/train_medians.pkl", "rb") as f:
        train_medians = pickle.load(f)
    with open(f"{DATA_DIR}/scale_cols.pkl", "rb") as f:
        scale_cols = pickle.load(f)

    feature_cols = [c for c in test_df.columns if c not in ['PatientID', 'SepsisLabel']]
    scale_col_indices = [feature_cols.index(c) for c in scale_cols]

    return model, test_seq, test_mask, test_labels, test_pids, device, scaler, train_medians, scale_cols, scale_col_indices


model, test_seq, test_mask, test_labels, test_pids, device, scaler, train_medians, scale_cols, _scale_col_indices = load_everything()
agent = HealthTwinAgent(urgent_threshold=0.7, soft_threshold=0.5, db_path=f"{DATA_DIR}/dashboard_alerts.db")
comm = CommunicationModule(db_path=f"{DATA_DIR}/dashboard_alerts.db")

KEY_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp']


def get_risk_scores(idx):
    with torch.no_grad():
        seq_t = torch.tensor(test_seq[idx], dtype=torch.float32).unsqueeze(0).to(device)
        mask_t = torch.tensor(test_mask[idx], dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(seq_t, mask_t)
        return torch.sigmoid(logits).cpu().numpy()[0]


def compute_events(risk_scores, agent):
    events, last_level = [], AlertLevel.NONE
    for t, score in enumerate(risk_scores):
        level = agent.classify(score)
        if level != last_level and level != AlertLevel.NONE:
            events.append({"t": t, "risk_score": float(score), "level": level})
        last_level = level
    return events, last_level


# _scale_col_indices (computed inside load_everything, where test_df is in scope)
# maps the 37 scaled "scale_cols" to their positions within the 73-length
# feature vector used in test_seq.
def unscale_row(full_row):
    """Inverse-transform the scaled subset of one timestep back to real units."""
    sub = np.array([full_row[i] for i in _scale_col_indices]).reshape(1, -1)
    real = scaler.inverse_transform(sub)[0]
    return dict(zip(scale_cols, real))


@st.cache_data
def get_current_status(_idx_placeholder, idx):
    """Cheap summary used for the ICU overview grid: last risk score + alert level."""
    valid_len = int(test_mask[idx].sum())
    if valid_len == 0:
        return 0.0, AlertLevel.NONE
    scores = get_risk_scores(idx)[:valid_len]
    level = agent.classify(scores[-1])
    return float(scores[-1]), level


septic_mask = test_labels.sum(axis=1) > 0
septic_ids = [test_pids[i] for i in range(len(test_pids)) if septic_mask[i]]
other_ids = [p for p in test_pids if p not in septic_ids]
overview_pool = septic_ids[:40]

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = overview_pool[0]

tab0, tab1, tab2, tab_live = st.tabs(
    ["ICU Overview", "Patient Twin", "Manual Vitals Entry", "Live Simulation"]
)

# ---------------- TAB 0: ICU Overview (multi-patient) ----------------
with tab0:
    st.header("Remote Monitoring Center")
    st.caption("All monitored patients, color-coded by current risk level. Click a patient to open their Digital Twin.")

    rows = []
    for pid in overview_pool:
        idx = test_pids.index(pid)
        risk, level = get_current_status(pid, idx)
        rows.append((pid, risk, level))

    rows.sort(key=lambda r: r[1], reverse=True)

    urgent_rows = [r for r in rows if r[2] == AlertLevel.URGENT]
    soft_rows = [r for r in rows if r[2] == AlertLevel.SOFT]
    stable_rows = [r for r in rows if r[2] == AlertLevel.NONE]

    def render_group(title, group_rows, color):
        if not group_rows:
            return
        st.subheader(f"{title} ({len(group_rows)})")
        cols = st.columns(4)
        for i, (pid, risk, level) in enumerate(group_rows):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{pid}**")
                    st.markdown(f":{color}[Risk: {risk:.2f}]")
                    if st.button("Open Twin", key=f"open_{pid}"):
                        st.session_state.selected_patient = pid
                        st.session_state["_jump_to_twin"] = True

    render_group("URGENT", urgent_rows, "red")
    render_group("SOFT NOTIFY", soft_rows, "orange")
    render_group("STABLE", stable_rows, "green")

# ---------------- TAB 1: Patient Twin (single patient) ----------------
with tab1:
    st.sidebar.header("Select Patient")
    options = overview_pool + other_ids[:30]
    default_idx = options.index(st.session_state.selected_patient) if st.session_state.selected_patient in options else 0
    patient_id = st.sidebar.selectbox("Patient ID", options, index=default_idx)
    st.session_state.selected_patient = patient_id

    idx = test_pids.index(patient_id)
    valid_len = int(test_mask[idx].sum())
    risk_scores = get_risk_scores(idx)[:valid_len]
    events, current_level = compute_events(risk_scores, agent)

    if f"logged_{patient_id}" not in st.session_state:
        for e in events:
            agent.log_alert(patient_id, e["risk_score"], e["level"])
            comm.trigger_communication(patient_id, e["level"])
        st.session_state[f"logged_{patient_id}"] = True

    # --- Patient Twin State card ---
    current_vitals = unscale_row(test_seq[idx][valid_len - 1])
    prev_vitals = unscale_row(test_seq[idx][max(valid_len - 2, 0)]) if valid_len > 1 else current_vitals

    st.subheader(f"Digital Twin — {patient_id}")
    level_color = {"urgent_alert": "red", "soft_notify": "orange", "none": "green"}[current_level.value]
    level_label = {"urgent_alert": "URGENT", "soft_notify": "SOFT NOTIFY", "none": "STABLE"}[current_level.value]

    with st.container(border=True):
        vc = st.columns(len(KEY_VITALS) + 2)
        for i, v in enumerate(KEY_VITALS):
            val = current_vitals.get(v, 0)
            prev = prev_vitals.get(v, val)
            arrow = "↑" if val > prev + 0.5 else ("↓" if val < prev - 0.5 else "→")
            vc[i].metric(v, f"{val:.1f} {arrow}")
        vc[-2].metric("Sepsis Risk", f"{risk_scores[-1]:.2f}")
        vc[-1].markdown(f"**Alert**  \n:{level_color}[{level_label}]")
        st.caption(f"Last updated — Hour {valid_len - 1}")

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Risk Trend")
        st.line_chart(pd.DataFrame({"Risk Score": risk_scores}, index=range(valid_len)))
        st.caption("Thresholds — Soft notify: 0.50 | Urgent alert: 0.70")

    with col2:
        st.subheader("Current Status")
        if current_level == AlertLevel.URGENT:
            st.error(f"URGENT — risk {risk_scores[-1]:.2f}")
        elif current_level == AlertLevel.SOFT:
            st.warning(f"MONITORING — risk {risk_scores[-1]:.2f}")
        else:
            st.success(f"STABLE — risk {risk_scores[-1]:.2f}")
        st.metric("ICU hours recorded", valid_len)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Alert History")
        history = agent.get_alert_history(patient_id)
        if history:
            st.dataframe(pd.DataFrame(history, columns=["Timestamp", "Risk Score", "Level", "Message"]), use_container_width=True)
        else:
            st.info("No alerts raised for this patient.")

    with col4:
        st.subheader("Remote Care Team")
        for sender, msg, ts in comm.get_messages(patient_id):
            st.chat_message("assistant" if sender == "system" else "user").write(msg)

        if current_level == AlertLevel.URGENT:
            ack_key = f"ack_{patient_id}"
            if st.button("Acknowledge Alert", key=ack_key):
                comm.send_message(patient_id, "clinician", "Alert acknowledged. Reviewing patient now.")
                st.rerun()
            link = comm.create_video_call(patient_id)
            st.link_button("Start Video Consultation (optional)", link)

        user_msg = st.text_input("Send a message to the care team", key="msg_input")
        if st.button("Send") and user_msg:
            comm.send_message(patient_id, "clinician", user_msg)
            st.rerun()

# ---------------- TAB 2: Manual Vitals Entry ----------------
with tab2:
    st.header("Manual Patient Entry — Update Digital Twin")
    st.caption("Enter current vitals to update this patient's twin and recompute risk instantly.")

    with st.form("manual_entry_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            hr = st.number_input("Heart Rate (HR)", value=float(train_medians.get('HR', 80.0)))
            o2sat = st.number_input("O2 Saturation", value=float(train_medians.get('O2Sat', 97.0)))
        with c2:
            temp = st.number_input("Temperature (°C)", value=float(train_medians.get('Temp', 37.0)))
            sbp = st.number_input("Systolic BP", value=float(train_medians.get('SBP', 120.0)))
        with c3:
            map_val = st.number_input("MAP", value=float(train_medians.get('MAP', 80.0)))
            dbp = st.number_input("Diastolic BP", value=float(train_medians.get('DBP', 70.0)))
        with c4:
            resp = st.number_input("Respiration Rate", value=float(train_medians.get('Resp', 16.0)))
            age = st.number_input("Age", value=60.0)

        submitted = st.form_submit_button("Update Twin & Check Risk")

    if submitted:
        manual_row = train_medians.reindex(scale_cols).fillna(0).copy()
        manual_row['HR'] = hr
        manual_row['O2Sat'] = o2sat
        manual_row['Temp'] = temp
        manual_row['SBP'] = sbp
        manual_row['MAP'] = map_val
        manual_row['DBP'] = dbp
        manual_row['Resp'] = resp
        if 'Age' in manual_row.index:
            manual_row['Age'] = age

        manual_scaled = scaler.transform(manual_row.values.reshape(1, -1))[0]

        n_feat = test_seq.shape[2]      # 73 (full feature vector the model expects)
        seq_len = test_seq.shape[1]
        full_row = np.zeros(n_feat, dtype=np.float32)
        for j, col_idx in enumerate(_scale_col_indices):
            full_row[col_idx] = manual_scaled[j]

        manual_seq = np.zeros((1, seq_len, n_feat), dtype=np.float32)
        manual_seq[0, 0] = full_row
        manual_mask = np.zeros((1, seq_len), dtype=np.float32)
        manual_mask[0, 0] = 1

        with torch.no_grad():
            seq_t = torch.tensor(manual_seq).to(device)
            mask_t = torch.tensor(manual_mask).to(device)
            logits = model(seq_t, mask_t)
            risk = torch.sigmoid(logits)[0, 0].item()

        level = agent.classify(risk)
        st.metric("Updated Twin — Risk Score", f"{risk:.2f}")
        if level.value == "urgent_alert":
            st.error(f"URGENT — {agent._generate_message(risk, level)}")
        elif level.value == "soft_notify":
            st.warning(f"{agent._generate_message(risk, level)}")
        else:
            st.success("No immediate concern based on entered vitals.")

# ---------------- TAB 3: Live Simulation (remote stream) ----------------
with tab_live:
    st.header("Simulated Remote Patient Stream")
    st.caption("Replays this patient's historical vitals hour-by-hour, as if streamed from a remote monitor.")

    live_patient = st.selectbox("Patient to simulate", septic_ids[:20], key="live_patient")
    speed = st.slider("Playback speed (seconds per hour)", 0.1, 2.0, 0.3)
    start_btn = st.button("Start Remote Stream")

    if start_btn:
        idx_live = test_pids.index(live_patient)
        valid_len_live = int(test_mask[idx_live].sum())
        risk_scores_live = get_risk_scores(idx_live)[:valid_len_live]

        stream_log = st.empty()
        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        alert_placeholder = st.empty()

        history = []
        last_level = AlertLevel.NONE
        log_lines = []

        for t in range(valid_len_live):
            history.append(risk_scores_live[t])
            level = agent.classify(risk_scores_live[t])

            log_lines.append(f"Hour {t} → received ✓")
            stream_log.code("\n".join(log_lines[-6:]))
            chart_placeholder.line_chart(pd.DataFrame({"Risk Score": history}))

            if level == AlertLevel.URGENT:
                status_placeholder.error(f"Twin Status: UPDATED — Risk {risk_scores_live[t]:.2f} — 🔴 URGENT")
            elif level == AlertLevel.SOFT:
                status_placeholder.warning(f"Twin Status: UPDATED — Risk {risk_scores_live[t]:.2f} — 🟠 SOFT NOTIFY")
            else:
                status_placeholder.success(f"Twin Status: UPDATED — Risk {risk_scores_live[t]:.2f} — 🟢 STABLE")

            if level != last_level and level != AlertLevel.NONE:
                msg = agent._generate_message(risk_scores_live[t], level)
                alert_placeholder.info(f"New alert at hour {t}: {msg}")
            last_level = level
            time.sleep(speed)

        st.success("Stream complete — full stay replayed.")
