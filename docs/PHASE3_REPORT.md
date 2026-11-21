# HealthTwin V4 - Phase 3 Implementation Report

## Scope completed

Phase 3 links the existing diabetes, cardiovascular and hypertension tabular models to registered patient Digital Twins. Kidney disease was removed from the project scope as requested.

### Patient-linked assessment flow

1. A user selects a registered patient on the Disease Prediction page.
2. The user enters the clinical inputs for one disease. The patient's age and sex are supplied from the profile only when the corresponding field is blank and the model uses that demographic field.
3. The service checks that the disease model has passed the existing real-data provenance, artifact and evaluation-report gate.
4. The disease's own model calculates its cross-sectional estimate. The sepsis model is never reused.
5. The assessment is saved with its disease, patient ID, twin hour, entered input values, assessment timestamp and result in SQLite.
6. The Digital Twin keeps separate assessment histories for diabetes, cardiovascular disease and hypertension. The patient timeline receives an assessment event.
7. The histories are restored from SQLite when a new PatientService instance loads the patient.

## Important boundaries

- These tabular models are baseline/cross-sectional estimates, not temporal forecasts.
- A result is saved to a patient's history only when the corresponding model passes the validated-model gate. Pipeline-test models may be used only in the existing developer test mode and are not saved as validated patient assessments.
- Missing tabular inputs continue to use the model's training-set imputation and are disclosed by the model response.
- These assessments do not automatically trigger alerts. The existing alert system remains tied to the sepsis stream and anomaly rules.
- The models are academic prototypes, not diagnostic tools or clinically validated decision systems.
- No new trained model artifacts or datasets were fabricated or added to the ZIP.

## Kidney removal

Removed kidney from the tabular model registry/specification, data download script, environment model-gate list, Prediction page, tests and documentation. The remaining tabular model set is diabetes, cardiovascular disease and hypertension.

## Verification

- `python -m compileall -q app.py src scripts tests`: passed.
- `python -m pytest -q tests/test_patient_platform.py tests/test_docs.py tests/test_config.py tests/test_phase1_gate.py`: 36 passed.
- `python -m pytest -q tests/test_phase2_tabular.py -k train_script_skips_missing_dataset`: 1 passed, 9 deselected.
- Full `tests/test_phase2_tabular.py` did not finish within the available execution window.
- Dashboard AppTests could not run in this verification environment because Streamlit is not installed here.
- `tests/test_forecast_ecg.py` collection could not run because wfdb is not installed here.

Run the full suite on the Windows project environment after installing `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

## Files changed

- `src/database/patient_store.py`
- `src/digital_twin/twin.py`
- `src/services/patient_service.py`
- `src/diseases/registry.py`
- `src/diseases/tabular.py`
- `src/simulation/session.py`
- `scripts/download_tabular.py`
- `scripts/train_tabular.py`
- `scripts/check_environment.py`
- `app.py`
- `tests/test_patient_platform.py`
- `tests/test_phase2_tabular.py`
- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/limitations_and_ethics.md`
- `docs/system_architecture.md`
- `docs/platform_guide.md`
