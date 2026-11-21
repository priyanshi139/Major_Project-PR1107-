import numpy as np
import pandas as pd
from src.digital_twin.models import Observation
from src.data_ingestion.sepsis_loader import FEATURES, VITALS

MODULE = "sepsis"


def build_observations(patient_df: pd.DataFrame, source="historical_dataset") -> list:
    """Real historical patient rows -> Observation list (chronological)."""
    ids = patient_df["subject_id"].unique()
    if len(ids) != 1:
        raise ValueError("Exactly one subject required")
    out = []
    for _, r in patient_df.sort_values("hour").iterrows():
        out.append(Observation(r["subject_id"], MODULE, int(r["hour"]),
                               {v: (None if pd.isna(r[v]) else float(r[v])) for v in FEATURES}, None, source))
    return out


def make_synthetic_observations(seed=42, n_hours=48, deterioration_hour=30):
    """SYNTHETIC demo data (NOT real patient data). Subject id contains 'SYNTHETIC'."""
    rng = np.random.default_rng(seed)
    out = []
    for h in range(n_hours):
        det = max(0, h - deterioration_hour)
        vals = {v: None for v in FEATURES}
        vals.update({"HR": 80 + rng.normal(0, 2) + 3.0 * det, "Resp": 16 + rng.normal(0, 0.8) + 0.8 * det,
                     "Temp": 36.8 + rng.normal(0, 0.1) + 0.12 * det, "O2Sat": 97 + rng.normal(0, 0.5) - 0.3 * det,
                     "SBP": 120 + rng.normal(0, 3) - 2.0 * det, "MAP": 85 + rng.normal(0, 2) - 1.5 * det,
                     "Age": 65.0, "ICULOS": float(h + 1), "HospAdmTime": -10.0})
        if rng.random() < 0.15:
            vals["HR"] = None                      # simulate missing measurements
        out.append(Observation("sepsis:SYNTHETIC_demo", MODULE, h, vals, None, "synthetic_demo"))
    return out
