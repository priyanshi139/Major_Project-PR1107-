"""SHAP explanations for patient-level tabular disease risk estimates.

Uses model-agnostic permutation SHAP on the training background saved with each
model artifact. Explanations describe model associations, not causes or clinical
importance. They are local to the supplied assessment and are not diagnoses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explain_tabular_prediction(model, values: dict, max_evals: int | None = None) -> dict:
    """Return local SHAP contributions for the calibrated positive-class score.

    Missing inputs are represented as NaN and imputed by the fitted pipeline,
    exactly as in inference. The training background is retained in the artifact.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError("SHAP is not installed. Run: python -m pip install shap") from exc

    clean, errors = model.clean(values)
    if errors:
        raise ValueError("; ".join(errors))
    if not clean:
        raise ValueError("Provide at least one valid input before requesting an explanation")

    row = model._row(clean)
    background = model.background
    if not isinstance(background, pd.DataFrame):
        background = pd.DataFrame(background, columns=model.features)
    background = background[model.features].copy()
    # Explain the same calibrated score returned to the user, not the raw model output.
    def score_fn(frame):
        frame = pd.DataFrame(frame, columns=model.features)
        raw = model.pipe.predict_proba(frame)[:, 1]
        if model.platt is not None:
            from src.diseases.tabular import _logit
            return model.platt.predict_proba(_logit(raw))[:, 1]
        return raw

    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(score_fn, masker, algorithm="permutation", feature_names=model.features)
    n_features = len(model.features)
    budget = int(max_evals or max(2 * n_features + 1, 100))
    explanation = explainer(row, max_evals=budget)
    contributions = np.asarray(explanation.values)
    if contributions.ndim == 2:
        contributions = contributions[0]
    base_values = np.asarray(explanation.base_values).reshape(-1)
    base = float(base_values[0])
    vals = row.iloc[0].to_dict()
    items = []
    for idx, feature in enumerate(model.features):
        shap_value = float(contributions[idx])
        items.append({
            "feature": feature,
            "label": next((f.label for f in model.spec.features if f.name == feature), feature),
            "shap_value": shap_value,
            "direction": "increases" if shap_value > 0 else "decreases" if shap_value < 0 else "no material change",
            "input_value": None if pd.isna(vals[feature]) else float(vals[feature]),
            "missing_input": bool(pd.isna(vals[feature])),
        })
    items.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    predicted = float(score_fn(row)[0])
    return {
        "disease": model.key,
        "model_name": model.name,
        "base_value": base,
        "predicted_score": predicted,
        "items": items,
        "method": "Model-agnostic permutation SHAP on the calibrated positive-class score",
        "max_evals": budget,
        "note": ("SHAP values describe how this model's inputs contribute to this individual score "
                 "relative to the saved training background. They are associations, not causal effects, "
                 "not clinical explanations, and not evidence that changing a feature will change health risk."),
    }
