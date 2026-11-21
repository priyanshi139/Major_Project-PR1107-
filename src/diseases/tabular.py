"""Disease-specific tabular risk models: diabetes, cardiovascular disease and hypertension.

Each disease has its OWN dataset, feature list, preprocessing, model, calibration, threshold, evaluation and artifact folder.
One row = one person, so a stratified random split is a patient-wise split (no person appears in two splits).
Protocol: split 70/15/15 -> model selection by 5-fold CV AUROC on TRAIN only -> Platt calibration + threshold on VALIDATION
-> TEST used once for reporting. Nothing is fitted on validation/test except the calibrator and threshold (validation only)."""
import json, platform, importlib.metadata as md
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import choose_threshold_max_f1, compute_metrics
from src.utils.provenance import PIPELINE_VERSION, now_utc, read_manifest, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "public_tabular"
DISCLAIMER = "Academic prototype. Risk score is NOT a diagnosis and is not clinically validated."


@dataclass(frozen=True)
class Feat:
    name: str
    label: str
    unit: str
    lo: float
    hi: float
    default: float
    kind: str = "float"          # float | int | bool


@dataclass
class DiseaseSpec:
    key: str
    name: str
    csv: str
    label: str
    features: list
    dataset: str
    task: str
    limitations: list
    zero_is_missing: list = field(default_factory=list)
    group_cols: dict = field(default_factory=dict)      # subgroup analysis: name -> (column in raw df)
    sex_col: str = None
    expected: tuple = None       # published (rows, positives); a mismatch is recorded as a warning, never hidden
    aliases: dict = field(default_factory=dict)   # alternative column names -> canonical name
    categorical: dict = field(default_factory=dict)  # column -> {text value: 0/1}
    note_extra: str = ""


PIMA = [Feat("Pregnancies", "Pregnancies", "count", 0, 20, 2, "int"), Feat("Glucose", "Plasma glucose (OGTT)", "mg/dL", 40, 300, 110),
        Feat("BloodPressure", "Diastolic blood pressure", "mm Hg", 30, 140, 72), Feat("SkinThickness", "Triceps skin-fold", "mm", 5, 100, 25),
        Feat("Insulin", "2-h serum insulin", "mu U/mL", 5, 900, 80), Feat("BMI", "Body-mass index", "kg/m2", 12, 70, 30),
        Feat("DiabetesPedigreeFunction", "Diabetes pedigree function", "", 0.05, 2.5, 0.4), Feat("Age", "Age", "years", 18, 100, 35, "int")]

FHS_COMMON = [Feat("male", "Sex (1 = male)", "", 0, 1, 0, "bool"), Feat("age", "Age", "years", 18, 100, 50, "int"),
              Feat("education", "Education level (1-4)", "", 1, 4, 2, "int"), Feat("currentSmoker", "Current smoker", "", 0, 1, 0, "bool"),
              Feat("cigsPerDay", "Cigarettes per day", "count", 0, 80, 0, "int"), Feat("prevalentStroke", "History of stroke", "", 0, 1, 0, "bool"),
              Feat("diabetes", "Diabetes", "", 0, 1, 0, "bool"), Feat("totChol", "Total cholesterol", "mg/dL", 80, 700, 200),
              Feat("BMI", "Body-mass index", "kg/m2", 12, 70, 26), Feat("heartRate", "Resting heart rate", "beats/min", 30, 160, 75),
              Feat("glucose", "Fasting glucose", "mg/dL", 40, 500, 85)]
FHS_BP = [Feat("sysBP", "Systolic blood pressure", "mm Hg", 70, 280, 125), Feat("diaBP", "Diastolic blood pressure", "mm Hg", 40, 160, 80),
          Feat("BPMeds", "On blood-pressure medication", "", 0, 1, 0, "bool"), Feat("prevalentHyp", "Diagnosed hypertension", "", 0, 1, 0, "bool")]
# Cardiovascular: all baseline risk factors incl. BP. Hypertension: BP readings and BP medication are EXCLUDED (they define the label).
CVD_FEATS = FHS_COMMON + FHS_BP
HTN_FEATS = FHS_COMMON



SPECS = {
    "diabetes": DiseaseSpec(
        "diabetes", "Type 2 Diabetes", "pima_diabetes.csv", "Outcome", PIMA,
        "Pima Indians Diabetes Database (NIDDK/UCI), 768 women aged >= 21 of Pima heritage",
        "Diabetes diagnosis within 5 years of the baseline examination (binary label as provided)",
        ["Only 768 women of a single ancestry: not representative of men or other populations.",
         "Zeros in glucose, blood pressure, skin-fold, insulin and BMI are physiologically impossible and are treated as missing.",
         "Small test set (about 115 people): wide confidence intervals."],
        zero_is_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"], expected=(768, 268)),
    "cardiovascular": DiseaseSpec(
        "cardiovascular", "Cardiovascular disease (10-year CHD)", "framingham.csv", "TenYearCHD", CVD_FEATS,
        "Framingham Heart Study teaching subset (4,240 participants), public mirror",
        "10-year risk of coronary heart disease from baseline risk factors",
        ["Teaching subset of a 1940s-70s US cohort: not representative of present-day or Indian populations.",
         "Missing values (glucose ~9%) are median-imputed with training medians.",
         "Predicts a 10-year outcome from one baseline exam; it is not a diagnostic test."],
        sex_col="male", expected=(4240, 644)),
    "hypertension": DiseaseSpec(
        "hypertension", "Hypertension (screening)", "framingham.csv", "prevalentHyp", HTN_FEATS,
        "Framingham Heart Study teaching subset (4,240 participants), public mirror",
        "Prevalent hypertension at baseline predicted from NON-blood-pressure risk factors (BP readings and BP medication excluded to avoid label leakage)",
        ["Cross-sectional screening estimate (is hypertension likely present?), not a prediction of future hypertension.",
         "Blood-pressure readings are deliberately excluded; a real BP measurement is always the proper test.",
         "Same cohort limitations as the cardiovascular model."],
        sex_col="male", expected=(4240, 1317)),

}


def load_dataset(key: str, raw_dir=None) -> pd.DataFrame:
    s = SPECS[key]
    p = Path(raw_dir or RAW_DIR) / s.csv
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/download_tabular.py")
    df = pd.read_csv(p, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=s.aliases)
    missing = [c for c in [f.name for f in s.features] + [s.label] if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name}: missing columns {missing}")
    if s.categorical:
        obj = df.select_dtypes(include="object").columns
        for c in obj:                                     # strip tabs/spaces; '?' = not recorded
            v = df[c].astype(str).str.strip().str.lower(); df[c] = v.mask(v.isin(["?", "nan", ""]))
        for c, mp in s.categorical.items():
            bad = set(df[c].dropna().unique()) - set(mp)
            if bad:
                raise ValueError(f"{p.name}: column {c} has unexpected values {sorted(bad)}")
            df[c] = df[c].map(mp)
        for f in s.features:
            if f.name not in s.categorical:
                df[f.name] = pd.to_numeric(df[f.name], errors="coerce")
    for c in s.zero_is_missing:
        df[c] = df[c].replace(0, np.nan)
    df = df[df[s.label].notna()].copy()
    return df.reset_index(drop=True)


def dataset_check(key: str, df: pd.DataFrame) -> dict:
    """Compare the loaded file with the published size/prevalence. A mismatch is a warning stored in the report."""
    s = SPECS[key]
    got = (int(len(df)), int(df[s.label].sum()))
    ok = s.expected is None or got == tuple(s.expected)
    return {"rows": got[0], "positives": got[1], "expected_rows_positives": list(s.expected) if s.expected else None, "matches_published": ok,
            "warning": None if ok else f"File has {got[0]} rows / {got[1]} positives but the published dataset has {s.expected[0]} / {s.expected[1]}. "
                                       "It may be a modified copy; results are labelled accordingly."}


def make_splits(y: pd.Series, seed=42, fracs=(0.70, 0.15, 0.15)) -> dict:
    """Stratified split of row indices (one row = one person => patient-wise)."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fracs[0], stratify=y.values, random_state=seed)
    va, te = train_test_split(rest, train_size=fracs[1] / (fracs[1] + fracs[2]), stratify=y.values[rest], random_state=seed)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)), "split overlap"
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(te.tolist())}


def candidates(seed=42):
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                              n_jobs=1, random_state=seed)),
        "gradient_boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                                      random_state=seed)),
    }


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def bootstrap_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def train_disease(key: str, artifacts_dir, reports_dir, seed=42, raw_dir=None, df=None, provenance=None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = SPECS[key]
    df = load_dataset(key, raw_dir) if df is None else df
    if provenance is None:                       # default: whatever the data folder's PROVENANCE.json says (else 'unverified')
        rd = Path(raw_dir or RAW_DIR)
        provenance = dict(read_manifest(rd))
        fp = rd / s.csv
        if fp.exists():
            provenance["file"], provenance["file_sha256"] = s.csv, sha256_file(fp)
    check = dataset_check(key, df)
    names = [f.name for f in s.features]
    X, y = df[names].astype(float), df[s.label].astype(int)
    splits = make_splits(y, seed)
    Xtr, ytr = X.iloc[splits["train"]], y.iloc[splits["train"]]
    Xva, yva = X.iloc[splits["val"]], y.iloc[splits["val"]]
    Xte, yte = X.iloc[splits["test"]], y.iloc[splits["test"]]

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cvres, fitted = {}, {}
    for n, m in candidates(seed).items():
        sc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        cvres[n] = {"cv_auroc_mean": float(sc.mean()), "cv_auroc_sd": float(sc.std())}
        fitted[n] = m.fit(Xtr, ytr)
    best = max(cvres, key=lambda n: cvres[n]["cv_auroc_mean"])          # selected on TRAIN cross-validation only
    model = fitted[best]

    raw_va, raw_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    platt = None
    if yva.sum() >= 10 and (len(yva) - yva.sum()) >= 10:                 # calibrator fitted on VALIDATION only
        platt = LogisticRegression(C=1e6, max_iter=500).fit(_logit(raw_va), yva)
    cal = (lambda r: platt.predict_proba(_logit(r))[:, 1]) if platt is not None else (lambda r: np.asarray(r))
    p_va, p_te = cal(raw_va), cal(raw_te)
    thr = choose_threshold_max_f1(yva.values, p_va)                      # threshold from VALIDATION
    results = {"val": compute_metrics(yva.values, p_va, thr), "test": compute_metrics(yte.values, p_te, thr)}
    results["test"]["auroc_ci95"] = bootstrap_ci(yte.values, p_te, roc_auc_score)
    results["test"]["auprc_ci95"] = bootstrap_ci(yte.values, p_te, average_precision_score)
    # all candidates on test, for transparency (NOT used for selection)
    others = {n: {"test_auroc": float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])),
                  "test_auprc": float(average_precision_score(yte, m.predict_proba(Xte)[:, 1]))} for n, m in fitted.items()}
    # subgroup analysis on test (sex / age band) where both classes exist with >= 10 rows
    sub = {}
    te_df = df.iloc[splits["test"]]
    age_col = "Age" if "Age" in te_df.columns else "age"
    groups = {"age<50": te_df[age_col] < 50, "age>=50": te_df[age_col] >= 50}
    if s.sex_col:
        groups.update({"female": te_df[s.sex_col] == 0, "male": te_df[s.sex_col] == 1})
    for g, mask in groups.items():
        m_ = mask.values
        if m_.sum() >= 10 and len(np.unique(yte.values[m_])) == 2:
            sub[g] = {"n": int(m_.sum()), "prevalence": float(yte.values[m_].mean()), "auroc": float(roc_auc_score(yte.values[m_], p_te[m_]))}
        else:
            sub[g] = {"n": int(m_.sum()), "note": "too few rows or one class only; not evaluated"}

    medians = Xtr.median().to_dict()
    background = Xtr.sample(min(100, len(Xtr)), random_state=seed)
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    meta = {"disease": key, "disease_name": s.name, "dataset": s.dataset, "task": s.task, "selected_model": best,
            "selection_rule": "highest 5-fold CV AUROC on the training split", "seed": seed, "threshold": thr,
            "threshold_rule": "maximum F1 on validation split (calibrated probabilities)",
            "calibration": "Platt scaling on validation split" if platt is not None else "none (too few validation positives)",
            "features": names, "n_rows": int(len(df)), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
            "split_sizes": {k: len(v) for k, v in splits.items()}, "limitations": s.limitations,
            "versions": {"python": platform.python_version(), **{p: md.version(p) for p in ["numpy", "pandas", "scikit-learn"]}},
            "disclaimer": DISCLAIMER, "trained_at": now_utc(), "pipeline_version": PIPELINE_VERSION,
            "provenance": provenance, "dataset_check": check}
    joblib.dump({"model": model, "name": best, "platt": platt, "threshold": thr, "features": names, "medians": medians,
                 "background": background, "meta": meta}, artifacts_dir / "model.joblib")
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (artifacts_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    warnings = [check["warning"]] if check["warning"] else []
    if results["test"]["auroc"] >= 0.99:
        warnings.append("Near-perfect test AUROC on a small public dataset: the classes are very easy to separate here. Do not read this as real-world performance.")
    report = {"meta": meta, "cv_selection": cvres, "other_candidates_test": others, "results": results, "subgroups_test": sub, "warnings": warnings}
    (reports_dir / f"{key}_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    try:
        _plots(key, s.name, model, platt, Xte, yte, reports_dir)
    except Exception as e:
        report["plot_error"] = f"{type(e).__name__}: {e}"
    return report


def _plots(key, name, model, platt, Xte, yte, reports_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    raw = model.predict_proba(Xte)[:, 1]
    p = platt.predict_proba(_logit(raw))[:, 1] if platt is not None else raw
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(yte, p); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray"); ax[0].set(title=f"{name}: ROC (test)", xlabel="FPR", ylabel="TPR")
    pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr); ax[1].axhline(yte.mean(), ls="--", c="gray"); ax[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
    fp, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile"); ax[2].plot(mp, fp, marker="o"); ax[2].plot([0, 1], [0, 1], "--", c="gray")
    ax[2].set(title="Calibration (test)", xlabel="Mean predicted", ylabel="Observed fraction")
    fig.tight_layout(); fig.savefig(Path(reports_dir) / f"{key}_curves.png", dpi=110); plt.close(fig)


class TabularRiskModel:
    """Inference for one disease. Input: dict of feature -> value (missing = None)."""

    def __init__(self, artifacts_dir):
        p = Path(artifacts_dir) / "model.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found: {p}. Run scripts/train_tabular.py")
        b = joblib.load(p)
        self.pipe, self.name, self.platt, self.threshold = b["model"], b["name"], b["platt"], b["threshold"]
        self.features, self.medians, self.background, self.meta = b["features"], b["medians"], b["background"], b["meta"]
        self.key, self.spec = self.meta["disease"], SPECS[self.meta["disease"]]

    def explain(self, values: dict, max_evals=None) -> dict:
        """Generate a local permutation-SHAP explanation for this assessment."""
        from src.diseases.explain import explain_tabular_prediction
        return explain_tabular_prediction(self, values, max_evals=max_evals)

    # ---- validation ----
    def clean(self, values: dict):
        specs = {f.name: f for f in self.spec.features}
        clean, errors = {}, []
        for k, v in (values or {}).items():
            if k not in specs:
                continue
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{specs[k].label}: '{v}' is not a number"); continue
            if x != x or not (specs[k].lo <= x <= specs[k].hi):
                errors.append(f"{specs[k].label} {x:g} {specs[k].unit} is outside the plausible range {specs[k].lo:g}-{specs[k].hi:g}"); continue
            clean[k] = x
        return clean, errors

    def _row(self, clean):
        return pd.DataFrame([{f: clean.get(f, np.nan) for f in self.features}])

    def _calibrate(self, raw):
        return float(self.platt.predict_proba(_logit(np.array([raw])))[:, 1][0]) if self.platt is not None else float(raw)

    def category(self, score):
        return "High" if score >= self.threshold else "Elevated" if score >= 0.5 * self.threshold else "Low"

    def predict(self, values: dict, min_fields=3) -> dict:
        clean, errors = self.clean(values)
        if errors:
            raise ValueError("; ".join(errors))
        if len(clean) < min_fields:
            raise ValueError(f"Provide at least {min_fields} of the {len(self.features)} inputs for a {self.spec.name} estimate")
        raw = float(self.pipe.predict_proba(self._row(clean))[:, 1][0])
        score = self._calibrate(raw)
        missing = [f for f in self.features if f not in clean]
        return {"disease": self.key, "disease_name": self.spec.name, "risk_score": score, "raw_score": raw, "threshold": self.threshold,
                "flagged": score >= self.threshold, "category": self.category(score), "model_name": self.name,
                "n_inputs_given": len(clean), "n_inputs_total": len(self.features), "missing_inputs": missing,
                "imputation_note": ("Missing inputs were replaced by training-set medians." if missing else None),
                "calibration": self.meta["calibration"], "limitations": self.spec.limitations, "disclaimer": DISCLAIMER}


def load_all_tabular(artifacts_dir) -> dict:
    """-> {disease_key: TabularRiskModel or None}. Never raises; a missing artifact just means 'unavailable'."""
    out, errs = {}, {}
    for k in SPECS:
        try:
            out[k] = TabularRiskModel(Path(artifacts_dir) / k)
        except Exception as e:
            out[k], errs[k] = None, str(e)
    return out
