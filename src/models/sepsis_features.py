"""Causal feature builder: every feature at hour t uses only rows with hour <= t of the same patient."""
import pandas as pd
from src.data_ingestion.sepsis_loader import VITALS

ID_COLS = ["subject_id", "hour", "SepsisLabel"]


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """X = output of SepsisPreprocessor.transform (sorted by subject_id, hour)."""
    X = X.reset_index(drop=True)
    g = X.groupby("subject_id", sort=False)
    roll = g[VITALS].rolling(6, min_periods=1)
    mean6 = roll.mean().reset_index(level=0, drop=True).reindex(X.index)
    std6 = roll.std().reset_index(level=0, drop=True).reindex(X.index).fillna(0.0)
    diff1 = g[VITALS].diff().fillna(0.0)
    feats = X.drop(columns=[c for c in ID_COLS if c in X.columns])
    feats = pd.concat([feats,
                       mean6.add_suffix("_roll6_mean"),
                       std6.add_suffix("_roll6_std"),
                       diff1.add_suffix("_diff1")], axis=1)
    return feats
