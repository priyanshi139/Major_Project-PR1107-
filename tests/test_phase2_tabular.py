"""Phase 2: separate per-disease tabular pipelines (diabetes, cardiovascular, hypertension) + their gates.
All data here are SYNTHETIC test files with the same columns as the public datasets - nothing in this file is a real-data result."""
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.diseases.tabular import SPECS, TabularRiskModel, train_disease, load_dataset, dataset_check
from src.utils.provenance import check_gate, write_manifest
from src.simulation.session import load_assets
from src.diseases.registry import build_registry
from tests.test_dashboard import cfg_in

ROOT = Path(__file__).resolve().parents[1]


def _bin(rng, z):
    return (rng.random(len(z)) < 1 / (1 + np.exp(-z))).astype(int)


def make_raw(d, n=700, seed=0):
    """Write synthetic CSVs with the public datasets' column names. Signal: label depends on glucose/BMI/age/BP/creatinine."""
    rng = np.random.default_rng(seed); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    g, bmi, age = rng.normal(120, 30, n).clip(50, 250), rng.normal(32, 6, n).clip(18, 60), rng.integers(21, 80, n)
    pd.DataFrame({"Pregnancies": rng.integers(0, 10, n), "Glucose": g, "BloodPressure": rng.normal(72, 10, n).clip(40, 120), "SkinThickness": rng.normal(25, 8, n).clip(5, 60),
                  "Insulin": rng.normal(100, 50, n).clip(10, 600), "BMI": bmi, "DiabetesPedigreeFunction": rng.random(n) + .1, "Age": age,
                  "Outcome": _bin(rng, (g - 120) / 25 + (bmi - 32) / 10)}).to_csv(d / "pima_diabetes.csv", index=False)
    sbp, dbp = rng.normal(130, 20, n).clip(90, 220), rng.normal(82, 11, n).clip(55, 130)
    chol, a2 = rng.normal(230, 40, n).clip(120, 400), rng.integers(33, 70, n)
    prev = _bin(rng, (bmi - 32) / 6 + (a2 - 50) / 15)
    fr = pd.DataFrame({"male": rng.integers(0, 2, n), "age": a2, "education": rng.integers(1, 5, n), "currentSmoker": rng.integers(0, 2, n), "cigsPerDay": rng.integers(0, 30, n),
                       "BPMeds": prev * rng.integers(0, 2, n), "prevalentStroke": 0, "prevalentHyp": prev, "diabetes": 0, "totChol": chol, "sysBP": sbp, "diaBP": dbp, "BMI": bmi,
                       "heartRate": rng.normal(75, 10, n).clip(45, 120), "glucose": g, "TenYearCHD": _bin(rng, (sbp - 130) / 20 + (a2 - 50) / 10 - 1.5)})
    fr.loc[rng.random(n) < .1, "glucose"] = np.nan
    fr.to_csv(d / "framingham.csv", index=False)
    return d


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    t = tmp_path_factory.mktemp("tab"); raw = make_raw(t / "raw")
    out = {}
    for k in SPECS:
        out[k] = train_disease(k, t / "art" / k, t / "rep", 42, raw_dir=raw, provenance={"data_origin": "synthetic_test", "source": "unit test"})
    return {"root": t, "raw": raw, "reports": out}


def test_each_disease_has_its_own_artifact_and_features(trained):
    t = trained["root"]
    assert {p.name for p in (t / "art").iterdir()} == set(SPECS)
    for k, rep in trained["reports"].items():
        m = rep["meta"]
        assert m["disease"] == k and m["features"] == [f.name for f in SPECS[k].features] and SPECS[k].label not in m["features"]
        assert (t / "art" / k / "model.joblib").exists() and (t / "rep" / f"{k}_metrics.json").exists()
    f = lambda k: set(trained["reports"][k]["meta"]["features"])
    assert not ({"sysBP", "diaBP", "BPMeds", "prevalentHyp"} & f("hypertension"))          # BP defines the hypertension label -> excluded
    assert {"sysBP", "diaBP"} <= f("cardiovascular") and "Glucose" in f("diabetes")
    assert "TenYearCHD" not in f("cardiovascular")


def test_splits_are_disjoint_and_report_is_complete(trained):
    for k, rep in trained["reports"].items():
        sp = json.loads((trained["root"] / "art" / k / "splits.json").read_text(encoding="utf-8"))
        a, b, c = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert not (a & b or a & c or b & c) and len(a | b | c) == rep["meta"]["n_rows"]
        t = rep["results"]["test"]
        assert {"auroc", "auprc", "precision", "recall", "f1", "confusion_matrix", "brier", "auroc_ci95"} <= set(t)
        assert sum(t["confusion_matrix"].values()) == t["n_rows"] and 0 <= t["auroc"] <= 1
        assert rep["meta"]["threshold_rule"].startswith("maximum F1 on validation")


def test_models_learn_real_signal_not_constants(trained):
    for k in ("diabetes",):                                  # synthetic labels have an actual dependence on these inputs
        assert trained["reports"][k]["results"]["test"]["auroc"] > 0.7
    m = TabularRiskModel(trained["root"] / "art" / "diabetes")
    lo = m.predict({"Glucose": 80, "BMI": 22, "Age": 30})["risk_score"]; hi = m.predict({"Glucose": 200, "BMI": 45, "Age": 60})["risk_score"]
    assert 0 <= lo < hi <= 1


def test_predict_validation_and_missing_values(trained):
    m = TabularRiskModel(trained["root"] / "art" / "cardiovascular")
    with pytest.raises(ValueError): m.predict({"age": 50})                                   # too few inputs
    with pytest.raises(ValueError): m.predict({"age": 50, "sysBP": 9999, "male": 1})          # implausible
    with pytest.raises(ValueError): m.predict({"age": "abc", "sysBP": 120, "male": 1})
    r = m.predict({"age": 55, "sysBP": 150, "male": 1, "totChol": ""})
    assert r["n_inputs_given"] == 3 and "totChol" in r["missing_inputs"] and r["imputation_note"] and r["disease"] == "cardiovascular"


def test_synthetic_trained_models_are_never_available(trained):
    for k in SPECS:
        g = check_gate(trained["root"] / "art" / k, trained["root"] / "rep" / f"{k}_metrics.json", allow_test=False)
        assert g["state"] == "pipeline_test" and not g["passed"] and "SYNTHETIC" in g["reason"]


def test_real_origin_passes_gate_only_with_matching_report(tmp_path):
    raw = make_raw(tmp_path / "raw")
    train_disease("diabetes", tmp_path / "a", tmp_path / "r", 1, raw_dir=raw, provenance={"data_origin": "real_public", "source": "x"})
    assert check_gate(tmp_path / "a", tmp_path / "r" / "diabetes_metrics.json")["passed"]
    (tmp_path / "r" / "diabetes_metrics.json").unlink()
    assert not check_gate(tmp_path / "a", tmp_path / "r" / "diabetes_metrics.json", allow_test=False)["passed"]
    # a data folder with no PROVENANCE.json gives 'unverified' by default
    train_disease("diabetes", tmp_path / "a2", tmp_path / "r2", 1, raw_dir=raw)
    assert check_gate(tmp_path / "a2", tmp_path / "r2" / "diabetes_metrics.json", allow_test=False)["state"] == "pipeline_test"


def test_dataset_check_flags_modified_files(trained):
    chk = trained["reports"]["diabetes"]["meta"]["dataset_check"]
    assert not chk["matches_published"] and "published" in chk["warning"]
    assert trained["reports"]["diabetes"]["warnings"]


def test_load_assets_and_registry_follow_the_gate(trained, tmp_path, monkeypatch):
    cfg = cfg_in(tmp_path); arts, reps = Path(cfg["paths"]["artifacts_dir"]), Path(cfg["paths"]["reports_dir"])
    import shutil; reps.mkdir(parents=True, exist_ok=True)
    for k in ("diabetes",):
        shutil.copytree(trained["root"] / "art" / k, arts / k); shutil.copy(trained["root"] / "rep" / f"{k}_metrics.json", reps / f"{k}_metrics.json")
    monkeypatch.delenv("HEALTHTWIN_ALLOW_TEST_MODELS", raising=False)
    a = load_assets(cfg); reg = build_registry(a)
    assert a["tabular"]["diabetes"] is None and a["tabular"]["cardiovascular"] is None
    assert reg["diabetes"].status == "unavailable" and "SYNTHETIC" in reg["diabetes"].reason and reg["cardiovascular"].status == "unavailable"
    monkeypatch.setenv("HEALTHTWIN_ALLOW_TEST_MODELS", "1")
    a = load_assets(cfg); reg = build_registry(a)
    assert a["tabular"]["diabetes"] is not None and reg["diabetes"].status == "pipeline_test"
    assert reg["cardiovascular"].status == "unavailable" and reg["sepsis"].status == "unavailable"        # sepsis model never reused
    # validated gate -> available
    a["gates"]["diabetes"] = {"passed": True, "state": "validated", "origin": "real_public", "trained_at": "t", "reason": ""}
    assert build_registry(a)["diabetes"].status == "available"


def test_train_script_skips_missing_dataset_and_never_fakes():
    src = (ROOT / "scripts" / "train_tabular.py").read_text(encoding="utf-8")
    assert "SKIPPED" in src and "download_tabular" in src


def test_prediction_page_shows_labelled_pipeline_test_model_and_predicts(trained, tmp_path, monkeypatch):
    import shutil, yaml
    from tests.test_dashboard import make_app
    cfg = cfg_in(tmp_path); arts, reps = Path(cfg["paths"]["artifacts_dir"]), Path(cfg["paths"]["reports_dir"]); reps.mkdir(parents=True, exist_ok=True)
    shutil.copytree(trained["root"] / "art" / "diabetes", arts / "diabetes"); shutil.copy(trained["root"] / "rep" / "diabetes_metrics.json", reps / "diabetes_metrics.json")
    monkeypatch.setenv("HEALTHTWIN_ALLOW_TEST_MODELS", "1")
    at = make_app(tmp_path, monkeypatch)
    at.radio(key="page").set_value("🔬 Prediction").run()
    assert not at.exception
    assert any("PIPELINE TEST MODEL" in e.value for e in at.error)
    at.text_input(key="tab_diabetes_Glucose").set_value("150"); at.text_input(key="tab_diabetes_BMI").set_value("33"); at.text_input(key="tab_diabetes_Age").set_value("45")
    at.button(key="tab_go_diabetes").click().run()
    assert not at.exception and any("Inputs given" in str(m.value) or "3/8" in str(m.value) for m in at.markdown)
    at.text_input(key="tab_diabetes_Glucose").set_value("9999"); at.button(key="tab_go_diabetes").click().run()
    assert any("outside the plausible range" in e.value for e in at.error)
    assert any("no model available" in i.value for i in at.info)       # cardiovascular / hypertension tabs: honest empty state
