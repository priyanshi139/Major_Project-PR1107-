# HealthTwin V4 — Phase 4: SHAP Explainability

## Scope implemented

- Added model-agnostic permutation SHAP explanations for the three tabular disease models: diabetes, cardiovascular disease, and hypertension.
- Explanations target the calibrated positive-class score returned by the prediction pipeline, rather than an unrelated raw score.
- The explainer uses the training-background sample saved in each model artifact. Missing values remain missing in the explained row and are handled by the fitted training pipeline's imputer.
- The Prediction page can show a local explanation after a tabular estimate when the disease model has passed the existing availability gate.
- Explanations show the baseline score, current score, feature contributions, input values, and direction of contribution.
- Added tests for explanation structure, score range, and missing-input handling.

## Important boundaries

SHAP is an explanation of model behaviour, not a causal method. A positive contribution means the feature moves this model's score above its background expectation for this input; it does not mean changing the feature will change the patient's health outcome. It is not a diagnosis or a clinically validated explanation.

The explanation is enabled only for validated tabular model status. The existing sepsis occlusion explanation is unchanged; SHAP for temporal sepsis sequences and transformer attribution are not part of this phase.

## Runtime requirements

`shap` is already declared in `requirements.txt`. If it is missing in a Windows environment, install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Verification

The new SHAP unit tests use a small deterministic toy model solely to test the explanation interface. They do not constitute clinical model evaluation. Run the complete project suite with:

```powershell
python -m pytest -q
```

Actual patient-facing explanations require the corresponding real-data model artifacts to be trained and pass the project's provenance/evaluation availability gate.
