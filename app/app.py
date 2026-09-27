import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import pickle
import sqlite3
import time
import sys, os

sys.path.append(os.getcwd())
from agent import HealthTwinAgent, AlertLevel
from communication import CommunicationModule

# Relative paths — everything lives inside the repo, no Google Drive dependency
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE_DIR, "..", "dataset", "healthtwin_sepsis_data", "preprocessed", "test_preprocessed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sepsis_transformer_small_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MEDIANS_PATH = os.path.join(BASE_DIR, "train_medians.pkl")
SCALE_COLS_PATH = os.path.join(BASE_DIR, "scale_cols.pkl")
DB_PATH = os.path.join(BASE_DIR, "alerts.db")

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


def build_sequences(df, feature_cols, label_col='SepsisLabel', patient_col='PatientID', max_len=60):
    grouped = df.groupby(patient_col)
    n_patients = grouped.ngroups
    n_features = len(feature_cols)
    sequences = np.zeros((n_patients, max_len, n_features), dtype=np.float32)
    masks = np.zeros((n_patients, max_len), dtype=np.float32)
    labels = np.zeros((n_patients, max_len), dtype=np.float32)
    patient_ids = []
    for i, (pid, group) in enumerate(grouped):
        group = group.sort_values('ICULOS')
        feats = group[feature_cols].values.astype(np.float32)
        label = group[label_col].values.astype(np.float32)
        seq_len = min(len(feats), max_len)
        sequences[i, :seq_len] = feats[:seq_len]
        labels[i, :seq_len] = label[:seq_len]
        masks[i, :seq_len] = 1
        patient_ids.append(pid)
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    return sequences, masks, labels, patient_ids


@st.cache_resource
def load_everything():
    device = torch.device('cpu')  # Streamlit Cloud has no GPU
    test_df = pd.read_csv(DATASET_CSV)
    feature_cols = [c for c in test_df.columns if c not in ['PatientID', 'SepsisLabel']]
    test_seq, test_mask, test_labels, test_pids = build_sequences(test_df, feature_cols, max_len=60)

    n_features = test_seq.shape[2]
    model = SmallSepsisTransformer(n_features=n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MEDIANS_PATH, "rb") as f:
        train_medians = pickle.load(f)
    with open(SCALE_COLS_PATH, "rb") as f:
        scale_cols = pickle.load(f)

    return model, test_seq, test_mask, test_labels, test_pids, device, scaler, train_medians, scale_cols


with st.spinner("Loading model and data (first run only)..."):
    model, test_seq, test_mask, test_labels, test_pids, device, scaler, train_medians, scale_cols = load_everything()

agent = HealthTwinAgent(urgent_threshold=0.7, soft_threshold=0.5, db_path=DB_PATH)
comm = CommunicationModule(db_path=DB_PATH)


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


septic_mask = test_labels.sum(axis=1) > 0
septic_ids = [test_pids[i] for i in range(len(test_pids)) if septic_mask[i]]
other_ids = [p for p in test_pids if p not in septic_ids]

tab1, tab2, tab_live = st.tabs(["Patient Monitoring", "Manual Vitals Entry", "Live Simulation"])

with tab1:
    st.sidebar.header("Select Patient")
    patient_id = st.sidebar.selectbox("Patient ID", septic_ids[:50] + other_ids[:50])

    idx = test_pids.index(patient_id)
    valid_len = int(test_mask[idx].sum())
    risk_scores = get_risk_scores(idx)[:valid_len]
    events, current_level = compute_events(risk_scores, agent)

    if f"logged_{patient_id}" not in st.session_state:
        for e in events:
            agent.log_alert(patient_id, e["risk_score"], e["level"])
            comm.trigger_communication(patient_id, e["level"])
        st.session_state[f"logged_{patient_id}"] = True

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Patient {patient_id} — Risk Trend")
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
        st.metric("Latest risk score", f"{risk_scores[-1]:.2f}")
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
        st.subheader("Doctor-Patient Communication")
        for sender, msg, ts in comm.get_messages(patient_id):
            st.chat_message("assistant" if sender == "system" else "user").write(msg)
        if current_level == AlertLevel.URGENT:
            link = comm.create_video_call(patient_id)
            st.link_button("Join Video Consultation", link)
        user_msg = st.text_input("Send a message to the care team", key="msg_input")
        if st.button("Send") and user_msg:
            comm.send_message(patient_id, "clinician", user_msg)
            st.rerun()

with tab2:
    st.header("Manual Patient Entry — Live Risk Check")
    st.caption("Enter current vitals for any patient to get an instant risk assessment.")

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

        submitted = st.form_submit_button("Check Risk")

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

        manual_scaled = scaler.transform(manual_row.values.reshape(1, -1))
        manual_seq = np.zeros((1, 60, len(scale_cols)), dtype=np.float32)
        manual_seq[0, 0] = manual_scaled
        manual_mask = np.zeros((1, 60), dtype=np.float32)
        manual_mask[0, 0] = 1

        with torch.no_grad():
            seq_t = torch.tensor(manual_seq).to(device)
            mask_t = torch.tensor(manual_mask).to(device)
            logits = model(seq_t, mask_t)
            risk = torch.sigmoid(logits)[0, 0].item()

        level = agent.classify(risk)
        st.metric("Predicted Risk Score", f"{risk:.2f}")
        if level.value == "urgent_alert":
            st.error(f"URGENT — {agent._generate_message(risk, level)}")
        elif level.value == "soft_notify":
            st.warning(f"{agent._generate_message(risk, level)}")
        else:
            st.success("No immediate concern based on entered vitals.")

with tab_live:
    st.header("Live Monitoring Simulation")
    st.caption("Replays this patient's ICU stay hour-by-hour, like a real bedside monitor.")

    live_patient = st.selectbox("Patient to simulate", septic_ids[:20], key="live_patient")
    speed = st.slider("Playback speed (seconds per hour)", 0.1, 2.0, 0.3)
    start_btn = st.button("Start Live Monitoring")

    if start_btn:
        idx_live = test_pids.index(live_patient)
        valid_len_live = int(test_mask[idx_live].sum())
        risk_scores_live = get_risk_scores(idx_live)[:valid_len_live]

        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        alert_placeholder = st.empty()

        history = []
        last_level = AlertLevel.NONE

        for t in range(valid_len_live):
            history.append(risk_scores_live[t])
            level = agent.classify(risk_scores_live[t])
            chart_placeholder.line_chart(pd.DataFrame({"Risk Score": history}))

            if level == AlertLevel.URGENT:
                status_placeholder.error(f"Hour {t} — URGENT — risk {risk_scores_live[t]:.2f}")
            elif level == AlertLevel.SOFT:
                status_placeholder.warning(f"Hour {t} — MONITORING — risk {risk_scores_live[t]:.2f}")
            else:
                status_placeholder.success(f"Hour {t} — STABLE — risk {risk_scores_live[t]:.2f}")

            if level != last_level and level != AlertLevel.NONE:
                msg = agent._generate_message(risk_scores_live[t], level)
                alert_placeholder.info(f"New alert at hour {t}: {msg}")
            last_level = level
            time.sleep(speed)

        st.success("Simulation complete — full stay replayed.")
