import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.diseases.explain import explain_tabular_prediction


def _fake_model():
    x = pd.DataFrame({"feature_a": np.linspace(0, 1, 80), "feature_b": np.linspace(1, 0, 80)})
    y = (x.feature_a > 0.5).astype(int)
    pipe = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression()).fit(x, y)
    spec = SimpleNamespace(features=[SimpleNamespace(name="feature_a", label="Feature A"),
                                     SimpleNamespace(name="feature_b", label="Feature B")])
    return SimpleNamespace(
        clean=lambda values: ({k: v for k, v in values.items() if v is not None}, []),
        _row=lambda values: pd.DataFrame([{k: values.get(k, np.nan) for k in ["feature_a", "feature_b"]}]),
        background=x.sample(20, random_state=3), features=["feature_a", "feature_b"],
        pipe=pipe, platt=None, spec=spec, key="test_disease", name="logistic_regression",
    )


def test_shap_explanation_has_local_contributions_and_metadata():
    result = explain_tabular_prediction(_fake_model(), {"feature_a": 0.8, "feature_b": 0.2}, max_evals=10)
    assert result["disease"] == "test_disease"
    assert 0 <= result["predicted_score"] <= 1
    assert len(result["items"]) == 2
    assert {x["feature"] for x in result["items"]} == {"feature_a", "feature_b"}
    assert all("shap_value" in x and "direction" in x for x in result["items"])
    assert "not causal" in result["note"]


def test_shap_explanation_uses_calibrated_model_score():
    model = _fake_model()
    result = explain_tabular_prediction(model, {"feature_a": 0.8, "feature_b": None}, max_evals=10)
    missing = next(x for x in result["items"] if x["feature"] == "feature_b")
    assert missing["missing_input"] is True
    assert missing["input_value"] is None
