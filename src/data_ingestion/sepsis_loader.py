"""Loader for PhysioNet/CinC Challenge 2019 .psv files (pipe-separated, one file per patient)."""
from pathlib import Path
import zipfile
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
        "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
FEATURES = VITALS + LABS + DEMOGRAPHICS
EXPECTED_COLUMNS = FEATURES + [LABEL]
NAMESPACE = "sepsis"


def make_subject_id(file_stem: str) -> str:
    """Dataset-namespaced subject id, e.g. 'sepsis:p000001'."""
    return f"{NAMESPACE}:{file_stem}"


def extract_zip(zip_path, dest_dir) -> Path:
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return dest_dir


def find_psv_files(root) -> list:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Sepsis raw folder not found: {root}. See README 'Phase 2' for download steps.")
    files = sorted(root.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"No .psv files found under {root}")
    return files


def load_patient_file(path) -> pd.DataFrame:
    """Load one patient. Adds subject_id, source_set, hour (row order = hourly time index)."""
    path = Path(path)
    df = pd.read_csv(path, sep="|")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")
    df = df[EXPECTED_COLUMNS].copy()
    df.insert(0, "hour", range(len(df)))
    df.insert(0, "subject_id", make_subject_id(path.stem))
    parent = path.parent.name.lower()
    df.insert(1, "source_set", "A" if "seta" in parent else ("B" if "setb" in parent else "unknown"))
    return df


def load_all(root, limit=None) -> pd.DataFrame:
    files = find_psv_files(root)
    if limit:
        files = files[:limit]
    frames = [load_patient_file(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    if out["subject_id"].str.split(":").str[0].nunique() != 1:
        raise ValueError("Mixed namespaces detected")
    return out
