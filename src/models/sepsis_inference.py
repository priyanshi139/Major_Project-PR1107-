"""Inference: takes the observation history available UP TO NOW for ONE patient, returns a risk score."""
from pathlib import Path
import joblib, pandas as pd
from src.data_ingestion.sepsis_loader import FEATURES
from src.models.sepsis_features import build_features
from src.models.train_sepsis import DISCLAIMER


class SepsisRiskModel:
    def __init__(self, artifacts_dir):
        d = Path(artifacts_dir)
        if not (d / "model.joblib").exists() or not (d / "preprocessor.joblib").exists():
            raise FileNotFoundError(f"Sepsis model artifacts not found in {d}. Run scripts/train_sepsis.py first.")
        self.pre = joblib.load(d / "preprocessor.joblib")
        bundle = joblib.load(d / "model.joblib")
        self.model, self.name = bundle["model"], bundle["name"]
        self.threshold, self.feature_names = bundle["threshold"], bundle["feature_names"]

    @staticmethod
    def validate(history: pd.DataFrame):
        if history is None or len(history) == 0:
            raise ValueError("History is empty")
        need = ["subject_id", "hour"] + FEATURES
        miss = [c for c in need if c not in history.columns]
        if miss:
            raise ValueError(f"History missing columns: {miss}")
        if history["subject_id"].nunique() != 1:
            raise ValueError("History must contain exactly one subject")
        if not history["hour"].is_monotonic_increasing or history["hour"].duplicated().any():
            raise ValueError("History hours must be strictly increasing")

    def predict(self, history: pd.DataFrame) -> dict:
        self.validate(history)
        X = self.pre.transform(history[["subject_id", "hour"] + FEATURES])
        F = build_features(X)[self.feature_names]
        score = float(self.model.predict_proba(F.iloc[[-1]])[:, 1][0])
        top = []
        if hasattr(self.model, "coef_"):   # linear model: coefficient * value (association, not causation)
            contrib = self.model.coef_[0] * F.iloc[-1].values
            for i in sorted(range(len(contrib)), key=lambda j: -abs(contrib[j]))[:3]:
                top.append({"feature": self.feature_names[i], "contribution": float(contrib[i])})
        return {"subject_id": history["subject_id"].iloc[0], "model_name": self.name,
                "risk_score": score, "top_factors": top, "threshold": self.threshold, "flagged": score >= self.threshold,
                "last_hour_used": int(history["hour"].iloc[-1]), "n_observations_used": int(len(history)),
                "disclaimer": DISCLAIMER}
