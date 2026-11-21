"""Validate a user-uploaded single-patient table (CSV or PSV) for the sepsis module."""
import re
import numpy as np
import pandas as pd
from src.data_ingestion.sepsis_loader import FEATURES, LABEL
from src.digital_twin.whatif import LIMITS

MAX_ROWS = 500


def validate_patient_upload(df: pd.DataFrame, name="upload"):
    """Returns (clean_df or None, errors, warnings). clean_df has subject_id, hour, all FEATURES, SepsisLabel if given."""
    errors, warnings = [], []
    if df is None or len(df) == 0:
        return None, ["The file has no rows."], warnings
    if len(df) > MAX_ROWS:
        return None, [f"Too many rows ({len(df)}); the limit is {MAX_ROWS} hourly rows."], warnings
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    known = [c for c in FEATURES if c in df.columns]
    if not known:
        return None, ["None of the expected columns were found. Expected some of: " + ", ".join(FEATURES[:10]) + ", ..."], warnings
    unknown = [c for c in df.columns if c not in FEATURES + ["hour", LABEL, "subject_id"]]
    if unknown:
        warnings.append(f"Ignored unknown columns: {unknown}")
    if "hour" in df.columns:
        h = pd.to_numeric(df["hour"], errors="coerce")
        if h.isna().any() or (h % 1 != 0).any():
            errors.append("Column 'hour' must contain whole numbers only.")
        elif (h < 0).any() or not h.is_monotonic_increasing or h.duplicated().any():
            errors.append("Column 'hour' must be non-negative, unique and strictly increasing.")
        hours = h.astype("Int64") if not errors else None
    else:
        warnings.append("No 'hour' column: row order is used as hours 0, 1, 2, ...")
        hours = pd.Series(range(len(df)))
    if errors:
        return None, errors, warnings
    out = pd.DataFrame({"hour": hours.astype(int).values})
    for c in FEATURES:
        if c in df.columns:
            raw = df[c]
            num = pd.to_numeric(raw, errors="coerce")
            bad = int((num.isna() & raw.notna() & (raw.astype(str).str.strip() != "")).sum())
            if bad:
                warnings.append(f"{c}: {bad} non-numeric value(s) treated as missing.")
            if c in LIMITS:
                lo, hi = LIMITS[c]
                oor = (num < lo) | (num > hi)
                if oor.any():
                    warnings.append(f"{c}: {int(oor.sum())} value(s) outside {lo}-{hi} treated as missing (possible error).")
                    num = num.mask(oor)
            out[c] = num.values
        else:
            out[c] = np.nan
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        warnings.append(f"{len(missing_cols)} expected columns absent (treated as never measured).")
    if not out[[c for c in LIMITS if c in out]].notna().any().any():
        return None, ["No usable vital-sign values (HR, Resp, MAP, SBP, O2Sat, Temp) after validation."], warnings
    if LABEL in df.columns:
        out[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int).values
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))[:30].strip("_") or "patient"
    out.insert(0, "subject_id", f"sepsis:UPLOADED_{slug}")
    return out, errors, warnings


def template_csv() -> str:
    cols = ["hour"] + FEATURES
    return ",".join(cols) + "\n" + ",".join(["0"] + [""] * len(FEATURES)) + "\n"
