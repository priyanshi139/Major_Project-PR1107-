# Platform Guide (how to use HealthTwin)

Run (Windows). **PowerShell:** `.\.venv\Scripts\Activate.ps1` then `streamlit run app.py`. **CMD:** `.venv\Scripts\activate.bat` then `streamlit run app.py`. The app opens at http://localhost:8501. No API key is needed.

1. **Overview** - if empty, create 3 synthetic demo patients (or register one under **Patients**).
2. **Patients / Profile** - register (ID `HT-0001`, `HT-0002`, ...), search, edit, choose multiple conditions, set baseline health values.
3. **Twin** - current vs previous state, baseline comparison, trends by source, forecasts (if trained), twin memory slider.
4. **Live** - type an observation (vitals required, labs optional) or start a *simulated* feed; every observation updates the twin, prediction, alerts and timeline.
5. **Prediction** - sepsis score, category, time, model info, history and occlusion explanation. Diabetes, cardiovascular and hypertension have separate models. Select a patient and enter that assessment's clinical values to save a validated result in the patient's per-disease Digital Twin history. Without a selected patient, it remains a stand-alone estimate. These are cross-sectional screening/risk estimates, not temporal forecasts or diagnoses.
6. **Timeline** - everything in order, filter by type and hours.
7. **What-if** - change a vital (e.g. MAP 65 to 75) and compare model scores; scenarios are saved separately and labelled simulated.
8. **Insights** - patient-specific findings with source and time.
9. **Alerts** - prioritised by severity; acknowledge, escalate, resolve; audit trail.
10. **Data** - import a public record, upload history, export CSVs, system status.

Add a new disease: train and validate a model on its own dataset, expose `predict(history_df)` returning a score dict, register it in `src/diseases/registry.py`, and give it its own namespace.
