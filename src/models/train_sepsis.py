import json, platform, importlib.metadata as md
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from src.preprocessing.sepsis_preprocessing import SepsisPreprocessor
from src.models.sepsis_features import build_features
from src.evaluation.metrics import choose_threshold_max_f1, compute_metrics
from src.data_ingestion.sepsis_loader import LABEL
from src.utils.logging_utils import get_logger
from src.utils.provenance import PIPELINE_VERSION, now_utc

log = get_logger("train_sepsis")
DISCLAIMER = "Academic prototype. Risk score is NOT a diagnosis and is not clinically validated."


def _versions():
    out = {"python": platform.python_version()}
    for p in ["numpy", "pandas", "scikit-learn", "joblib"]:
        out[p] = md.version(p)
    return out


def run_training(df: pd.DataFrame, splits: dict, artifacts_dir, reports_dir, seed=42, provenance=None):
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    part = {k: df[df["subject_id"].isin(v)] for k, v in splits.items()}
    for k, v in part.items():
        if v.empty:
            raise ValueError(f"Split '{k}' is empty")

    pre = SepsisPreprocessor().fit(part["train"])          # fit on TRAIN only
    data = {}
    for k, v in part.items():
        Xt = pre.transform(v)
        data[k] = (build_features(Xt), Xt[LABEL].values)
    feature_names = list(data["train"][0].columns)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=seed),
    }
    results, fitted = {}, {}
    for name, model in candidates.items():
        log.info("Training %s ...", name)
        model.fit(*data["train"])
        fitted[name] = model
        val_p = model.predict_proba(data["val"][0])[:, 1]
        thr = choose_threshold_max_f1(data["val"][1], val_p)      # threshold from VALIDATION
        test_p = model.predict_proba(data["test"][0])[:, 1]
        results[name] = {"val": compute_metrics(data["val"][1], val_p, thr),
                         "test": compute_metrics(data["test"][1], test_p, thr),
                         "threshold_source": "max F1 on validation set"}
        t = results[name]["test"]
        log.info("%s TEST: AUROC=%.3f AUPRC=%.3f (random=%.3f) P=%.3f R=%.3f F1=%.3f",
                 name, t["auroc"], t["auprc"], t["auprc_random_baseline"], t["precision"], t["recall"], t["f1"])

    best = max(results, key=lambda n: results[n]["val"]["auprc"])   # selected on VALIDATION AUPRC
    thr = results[best]["val"]["threshold"]
    joblib.dump(pre, artifacts_dir / "preprocessor.joblib")
    joblib.dump({"model": fitted[best], "name": best, "threshold": thr,
                 "feature_names": feature_names}, artifacts_dir / "model.joblib")
    meta = {"selected_model": best, "selection_rule": "highest validation AUPRC", "seed": seed,
            "threshold": thr, "n_features": len(feature_names), "versions": _versions(),
            "split_patients": {k: len(v) for k, v in splits.items()},
            "label_note": "SepsisLabel as provided by the PhysioNet 2019 dataset (row-level, hourly).",
            "trained_at": now_utc(), "pipeline_version": PIPELINE_VERSION,
            "n_patients": int(df["subject_id"].nunique()), "n_rows": int(len(df)),
            "provenance": provenance or {"data_origin": "unverified", "source": "caller did not state the data origin"},
            "disclaimer": DISCLAIMER}
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    report = {"meta": meta, "results": results}
    (reports_dir / "sepsis_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    try:
        _plots(fitted, data, reports_dir)
    except Exception as e:  # plots are optional
        log.warning("Plot generation skipped: %s", e)
    log.info("Selected model: %s | threshold=%.4f", best, thr)
    return report


def _plots(fitted, data, reports_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    Xte, yte = data["test"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for n, m in fitted.items():
        p = m.predict_proba(Xte)[:, 1]
        pr, rc, _ = precision_recall_curve(yte, p); ax[0].plot(rc, pr, label=n)
        fpr, tpr, _ = roc_curve(yte, p); ax[1].plot(fpr, tpr, label=n)
        from sklearn.calibration import calibration_curve
        fp, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile"); ax[2].plot(mp, fp, marker="o", label=n)
    ax[0].axhline(yte.mean(), ls="--", c="gray"); ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].plot([0, 1], [0, 1], "--", c="gray"); ax[1].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax[2].plot([0, 1], [0, 1], "--", c="gray"); ax[2].set(title="Calibration (test)", xlabel="Mean predicted", ylabel="Fraction positive")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(reports_dir / "sepsis_curves.png", dpi=120); plt.close(fig)
