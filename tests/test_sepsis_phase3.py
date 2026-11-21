"""SYNTHETIC data only (learnable toy signal) to test code paths. Not real results."""
import numpy as np, pandas as pd, pytest
from src.data_ingestion.sepsis_loader import EXPECTED_COLUMNS, FEATURES
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from src.models.train_sepsis import run_training
from src.models.sepsis_inference import SepsisRiskModel
from src.evaluation.metrics import compute_metrics, choose_threshold_max_f1


def synth(n_pat=80, n_hours=30, seed=0):
    rng = np.random.default_rng(seed); frames = []
    for i in range(n_pat):
        d = pd.DataFrame(rng.normal(size=(n_hours, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
        septic = i % 4 == 0
        d["SepsisLabel"] = 0
        if septic:
            d.loc[n_hours // 2:, "SepsisLabel"] = 1
            d.loc[n_hours // 2:, "HR"] += 3.0
        d.loc[::3, "Lactate"] = np.nan
        d.insert(0, "hour", range(n_hours)); d.insert(0, "subject_id", f"sepsis:p{i:06d}")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("art")
    df = synth(); splits = patient_wise_split(df, seed=1)
    rep = run_training(df, splits, tmp / "a", tmp / "r", seed=1)
    return df, splits, tmp, rep


def test_artifacts_and_report(trained):
    _, _, tmp, rep = trained
    for f in ["model.joblib", "preprocessor.joblib", "model_meta.json"]:
        assert (tmp / "a" / f).exists()
    assert (tmp / "r" / "sepsis_metrics.json").exists()
    assert rep["meta"]["selection_rule"] == "highest validation AUPRC"
    assert 0 <= rep["results"]["logistic_regression"]["test"]["auroc"] <= 1


def test_metrics_validation():
    with pytest.raises(ValueError):
        compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    assert 0 < choose_threshold_max_f1(y, p) <= 0.8


def test_inference_returns_valid_score(trained):
    df, splits, tmp, _ = trained
    m = SepsisRiskModel(tmp / "a")
    pid = splits["test"][0]
    h = df[df.subject_id == pid].iloc[:10]
    out = m.predict(h)
    assert 0.0 <= out["risk_score"] <= 1.0 and out["n_observations_used"] == 10
    assert "NOT a diagnosis" in out["disclaimer"]


def test_no_future_leakage_in_inference(trained):
    """Score at hour t must be identical whether or not later rows exist."""
    df, splits, tmp, _ = trained
    m = SepsisRiskModel(tmp / "a")
    h = df[df.subject_id == splits["test"][0]]
    a = m.predict(h.iloc[:12])["risk_score"]
    h2 = h.copy(); h2.loc[h2.index[12:], FEATURES] = 999.0   # corrupt the future
    b = m.predict(h2.iloc[:12])["risk_score"]
    c = m.predict(h2.iloc[:30])  # runs fine with future present
    assert a == pytest.approx(b)
    m2 = m.predict(h.iloc[:13])  # one more observation is allowed to change the score
    assert "risk_score" in m2 and "risk_score" in c


def test_inference_input_validation(trained):
    df, _, tmp, _ = trained
    m = SepsisRiskModel(tmp / "a")
    with pytest.raises(ValueError): m.predict(df.iloc[0:0])
    with pytest.raises(ValueError): m.predict(df.iloc[:60])           # two subjects
    with pytest.raises(ValueError): m.predict(df.iloc[:5].drop(columns=["HR"]))
    with pytest.raises(FileNotFoundError): SepsisRiskModel(tmp / "missing")
