from pathlib import Path


def list_records(records_dir) -> list:
    d = Path(records_dir)
    return sorted(p.stem for p in d.glob("*.hea")) if d.exists() else []


def load_record(records_dir, record: str, lead="MLII") -> dict:
    import wfdb
    base = Path(records_dir) / record
    for ext in (".hea", ".dat", ".atr"):
        if not base.with_suffix(ext).exists():
            raise FileNotFoundError(f"Missing {base.with_suffix(ext)}. Run scripts/download_ecg.py")
    r = wfdb.rdrecord(str(base)); a = wfdb.rdann(str(base), "atr")
    idx = r.sig_name.index(lead) if lead in r.sig_name else 0
    return {"record": record, "signal": r.p_signal[:, idx].astype(float), "fs": int(r.fs),
            "samples": a.sample.astype(int), "symbols": list(a.symbol), "lead": r.sig_name[idx]}
