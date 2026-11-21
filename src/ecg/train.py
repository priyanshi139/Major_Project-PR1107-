import json, platform, importlib.metadata as md
from src.utils.provenance import PIPELINE_VERSION, now_utc, read_manifest
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from src.ecg.beats import CLASSES, extract_beats, feature_matrix

# Inter-patient record split commonly used in the ECG literature (de Chazal et al.): DS1 for training, DS2 for testing.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
DISCLAIMER = "Academic prototype. Beat-level classification of MIT-BIH records; not a diagnostic tool."


def split_records(available) -> dict:
    av = set(available)
    ds1 = [r for r in DS1 if r in av]
    val = ds1[::5]
    train = [r for r in ds1 if r not in val]
    test = [r for r in DS2 if r in av]
    splits = {"train": train, "val": val, "test": test}
    s = [set(v) for v in splits.values()]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2]), "record overlap"
    return splits


def multiclass_metrics(y_true, y_pred) -> dict:
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    present = sup > 0
    return {"per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
                          for i, c in enumerate(CLASSES)},
            "macro_f1_present_classes": float(f[present].mean()) if present.any() else 0.0,
            "balanced_accuracy": float(r[present].mean()) if present.any() else 0.0,
            "accuracy_not_headline": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
            "confusion_matrix": {"labels": CLASSES, "matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
            "n_beats": int(len(y_true))}


def _stack(beats_by_record, records):
    parts = [beats_by_record[r] for r in records if r in beats_by_record and len(beats_by_record[r]["labels"])]
    if not parts:
        return np.zeros((0, 106)), np.array([])
    X = np.vstack([feature_matrix(b["waves"], b["rr"]) for b in parts])
    return X, np.concatenate([b["labels"] for b in parts])


def run_training_from_beats(beats_by_record, artifacts_dir, reports_dir, seed=42, max_n_train=20000, provenance=None):
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    splits = split_records(beats_by_record.keys())
    Xtr, ytr = _stack(beats_by_record, splits["train"]); Xva, yva = _stack(beats_by_record, splits["val"])
    Xte, yte = _stack(beats_by_record, splits["test"])
    if len(ytr) == 0 or len(yva) == 0 or len(yte) == 0:
        raise ValueError("Need beats in train, val and test records (DS1/DS2). Download more records.")
    rng = np.random.default_rng(seed)
    idx_n = np.where(ytr == "N")[0]
    if len(idx_n) > max_n_train:                                       # cap the dominant class for speed
        drop = rng.choice(idx_n, len(idx_n) - max_n_train, replace=False)
        keep = np.setdiff1d(np.arange(len(ytr)), drop); Xtr, ytr = Xtr[keep], ytr[keep]
    models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")),
              "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=seed)}
    results = {}
    for n, m in models.items():
        m.fit(Xtr, ytr)
        results[n] = {"val": multiclass_metrics(yva, m.predict(Xva)), "test": multiclass_metrics(yte, m.predict(Xte))}
    results["baseline_always_N"] = {"val": multiclass_metrics(yva, np.array(["N"] * len(yva))),
                                    "test": multiclass_metrics(yte, np.array(["N"] * len(yte)))}
    best = max(models, key=lambda n: results[n]["val"]["macro_f1_present_classes"])
    joblib.dump({"model": models[best], "name": best, "classes": CLASSES}, artifacts_dir / "model.joblib")
    meta = {"selected_model": best, "selection_rule": "highest validation macro-F1", "seed": seed,
            "splits": splits, "n_train_beats_after_cap": int(len(ytr)), "max_n_train_N_beats": max_n_train,
            "train_class_counts": {c: int((ytr == c).sum()) for c in CLASSES},
            "python": platform.python_version(), "scikit-learn": md.version("scikit-learn"), "disclaimer": DISCLAIMER,
            "trained_at": now_utc(), "pipeline_version": PIPELINE_VERSION,
            "provenance": provenance or {"data_origin": "unverified", "source": "caller did not state the data origin"}}
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (reports_dir / "ecg_metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    return {"meta": meta, "results": results}


def run_ecg_training(records_dir, artifacts_dir, reports_dir, seed=42, max_n_train=20000):
    from src.ecg.loader import list_records, load_record
    recs = [r for r in list_records(records_dir) if r in DS1 + DS2]
    if not recs:
        raise FileNotFoundError(f"No usable MIT-BIH records in {records_dir}. Run scripts/download_ecg.py")
    beats = {}
    for r in recs:
        rec = load_record(records_dir, r)
        beats[r] = extract_beats(rec["signal"], rec["samples"], rec["symbols"], rec["fs"])
    return run_training_from_beats(beats, artifacts_dir, reports_dir, seed, max_n_train, provenance=read_manifest(records_dir))
