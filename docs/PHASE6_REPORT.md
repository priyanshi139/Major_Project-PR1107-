# HealthTwin V4 — Phase 6: Patient-specific longitudinal Twin analytics

## Scope implemented

- Added a descriptive, per-vital trend summary to the existing `DigitalTwin`.
- Trend calculations use only observations already received by that patient's Twin.
- Missing measurements are ignored; each variable requires at least three measured points across distinct observation hours.
- Reports latest measured value, linear slope per hour, direction, point count and observed time range.
- Explicitly marks the result as `clinical_forecast: false`; this is descriptive analytics, not a clinically validated forecast.
- Added patient-specific baseline maturity metadata (`not_started`, `building`, `established`) to the patient snapshot.
- Exposes both additions through the existing `PatientService.snapshot` response.
- The trend is derived from persisted observations, so it is reconstructed after application/database restart without storing duplicate derived state.

## Verification performed

Command:

```powershell
python -m pytest -q tests/test_phase6_twin.py tests/test_patient_platform.py tests/test_phase5_transformer.py
```

Result in the available execution environment: **23 passed**.

The full test suite could not be collected in this environment because the optional `wfdb` dependency is not installed (`tests/test_forecast_ecg.py` imports it). This is an environment/dependency limitation, not a passing full-suite result. Install project requirements and rerun the full suite on the target Windows environment.

## Limitations

- This phase does not train new disease models.
- The short-window linear trend is descriptive only; it is not a medical forecast and must not be used for clinical decisions.
- Existing tabular models remain cross-sectional. Their assessments are not reinterpreted as temporal predictions.
- No real PhysioNet, CGM, longitudinal blood-pressure or ECG model training is claimed by this phase.
- Existing Phase 1–5 real-data training gates remain in force.
