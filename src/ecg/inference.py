from pathlib import Path
import joblib, numpy as np
from src.ecg.beats import extract_beats, feature_matrix


class EcgBeatModel:
    def __init__(self, artifacts_dir):
        p = Path(artifacts_dir) / "model.joblib"
        if not p.exists():
            raise FileNotFoundError(f"ECG model not found: {p}. Run scripts/train_ecg.py")
        b = joblib.load(p); self.model, self.name = b["model"], b["name"]

    def classify_record(self, rec: dict) -> dict:
        beats = extract_beats(rec["signal"], rec["samples"], rec["symbols"], rec["fs"])
        if len(beats["labels"]) == 0:
            return {"samples": np.array([], int), "true": np.array([]), "pred": np.array([])}
        pred = self.model.predict(feature_matrix(beats["waves"], beats["rr"]))
        return {"samples": beats["samples"], "true": beats["labels"], "pred": pred}
