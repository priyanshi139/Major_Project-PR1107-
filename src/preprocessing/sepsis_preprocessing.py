"""Leakage-safe preprocessing for the sepsis module.

Rules:
 - Splits are by patient (subject_id), never by row.
 - Forward-fill is applied within each patient only (uses past values, never future).
 - Imputation medians and scaling statistics are fit on TRAIN patients only.
"""
import numpy as np
import pandas as pd
from src.data_ingestion.sepsis_loader import FEATURES, LABEL


def patient_outcomes(df: pd.DataFrame) -> pd.Series:
    """1 if the patient ever has SepsisLabel==1, else 0 (used only to stratify the split)."""
    return df.groupby("subject_id")[LABEL].max()


def patient_wise_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42, stratify=True):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    out = patient_outcomes(df)
    rng = np.random.default_rng(seed)
    groups = [out[out == 1].index.to_numpy(), out[out == 0].index.to_numpy()] if stratify \
        else [out.index.to_numpy()]
    splits = {"train": [], "val": [], "test": []}
    for ids in groups:
        ids = np.array(sorted(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * train_frac)), int(round(n * val_frac))
        splits["train"] += list(ids[:n_tr])
        splits["val"] += list(ids[n_tr:n_tr + n_va])
        splits["test"] += list(ids[n_tr + n_va:])
    splits = {k: sorted(v) for k, v in splits.items()}
    check_split_integrity(splits)
    return splits


def check_split_integrity(splits: dict):
    sets = {k: set(v) for k, v in splits.items()}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"Patient overlap between {a} and {b}")
    return True


def causal_forward_fill(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Forward-fill within each patient, ordered by hour. Never looks at future rows."""
    cols = cols or FEATURES
    df = df.sort_values(["subject_id", "hour"]).copy()
    df[cols] = df.groupby("subject_id")[cols].ffill()
    return df


class SepsisPreprocessor:
    """Fit on train only. transform() is then applied to train/val/test alike."""

    def __init__(self, add_missing_indicators=True, max_missing_rate=0.999):
        self.add_missing_indicators = add_missing_indicators
        self.max_missing_rate = max_missing_rate
        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.indicator_cols_ = []
        self.feature_names_out_ = None

    def fit(self, train_df: pd.DataFrame):
        if train_df.empty:
            raise ValueError("Cannot fit on empty training data")
        raw = train_df[FEATURES]
        if self.add_missing_indicators:
            rate = raw.isna().mean()
            self.indicator_cols_ = [c for c in FEATURES if 0 < rate[c] <= self.max_missing_rate]
        filled = causal_forward_fill(train_df)[FEATURES]
        self.medians_ = filled.median().fillna(0.0)
        imputed = filled.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.stds_ = imputed.std().replace(0, 1.0).fillna(1.0)
        self.feature_names_out_ = list(FEATURES) + [f"{c}_missing" for c in self.indicator_cols_]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor not fitted")
        missing_cols = [c for c in FEATURES if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input missing columns: {missing_cols}")
        df = df.sort_values(["subject_id", "hour"]).copy()
        indicators = {f"{c}_missing": df[c].isna().astype(np.int8) for c in self.indicator_cols_}
        df = causal_forward_fill(df)
        X = df[FEATURES].fillna(self.medians_)
        X = (X - self.means_) / self.stds_
        for k, v in indicators.items():
            X[k] = v.loc[X.index].values
        X.insert(0, "hour", df["hour"].values)
        X.insert(0, "subject_id", df["subject_id"].values)
        if LABEL in df.columns:
            X[LABEL] = df[LABEL].values
        return X.reset_index(drop=True)
