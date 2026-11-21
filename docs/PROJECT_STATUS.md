# Project Status

| Area | Status |
|---|---|
| Sepsis pipeline (download, split, preprocessing, training, inference, evaluation) | Trained on 6,000 real PhysioNet/CinC 2019 patients (patient-wise 4200/900/900 split); gate = validated. Test AUROC about 0.77 but AUPRC only about 0.06 at 1.8% prevalence: weak, not clinically validated. |
| Model availability gate + data provenance (Phase 1) | Done (tested): a model is 'available' only with real-data provenance, a loadable artifact and a matching evaluation report |
| Digital Twin per patient (baseline, memory, data quality, transitions, restore from DB) | Done (tested) |
| Patient registration, profiles (multiple conditions), search, edit, delete | Done (tested) |
| Observation entry with all 34 measurable model inputs, validation, timestamps, persistence | Done (tested) |
| Simulated live monitoring (labelled) + replay of public records | Done (tested) |
| Prediction page (ID, disease, score, category, time, model info, history, explanation) | Done (tested) |
| Automatic alerts: 4 rule types, severity, cooldown, acknowledge/escalate/resolve, audit trail | Done (tested) |
| Patient timeline with filters and CSV export | Done (tested) |
| What-if: current-state scenario using the real model + trajectory scenario; stored separately | Done (tested) |
| AI insights: evidence-based with source and time; occlusion explanation | Done (tested); tabular SHAP is implemented in Phase 4 |
| Forecasting vs persistence baseline | Trained on real data; gate = validated. Beats persistence by a small margin on most vitals; two horizon-1 targets (O2Sat, Temp) are disabled by the project's own rule |
| ECG module (MIT-BIH, record-level, separate) | Trained on 48 real records; gate = validated. Test macro-F1 about 0.55 (always-Normal baseline 0.24); supraventricular (S) beats are poorly detected |
| Diabetes, cardiovascular, hypertension: separate dataset, training, evaluation and inference per disease (`src/diseases/tabular.py`) | Code done and tested on synthetic files. Trained and evaluated with `scripts/train_tabular.py`; the final ZIP ships the three trained artifacts and their reports (re-run the script to reproduce). Phase 3 links validated tabular assessments to each patient twin. |
| Data upload, export, system status | Done (tested) |
| Authentication / roles | Done in the FastAPI backend (JWT, PBKDF2, patient/doctor/admin, audit log, login throttle). The Streamlit app itself has no login |
| External validation, clinical validation | NOT done |
| Patient-linked tabular disease assessments | Implemented and unit-tested: separate validated model per disease, patient ID, timestamp/twin-hour association, persisted history and timeline event. Requires locally trained, validated artifacts; no automatic alerting for these screening estimates. |
| Phase 4: SHAP explanations for validated tabular disease estimates | Implemented: model-agnostic permutation SHAP on calibrated scores, training-background reference, Prediction-page display and unit tests. Requires locally trained artifacts and SHAP dependency. Sepsis occlusion remains unchanged; explanations are not causal or clinically validated. |
| Phase 5: causal Transformer + LSTM sequence training pipeline | Implemented and unit-tested: patient-local hourly windows, causal attention mask, train-only preprocessing, patient-wise splits, validation model selection and held-out test metrics. Trained on the real 6,000-patient subset; the LSTM was selected on validation AUPRC and neither sequence model clearly beats the simple row-wise baseline on the same windows. Current metrics are hourly-window metrics, not clinically validated patient-level lead-time utility. |


| Phase 6: patient-specific longitudinal Twin analytics | Implemented in this backup: per-patient baseline maturity metadata and descriptive measured-vital trend summaries; trend calculation uses only arrived observations and requires at least 3 measured time points. Trends are descriptive, not clinical forecasts. Unit/integration tests cover insufficient history, missing values, direction/slope and database restoration. |
| Phase 7: agentic alert layer + FastAPI | Implemented and tested: deterministic triage with evidence (value, source, time, data kind), duplicate-proposal suppression, proposals need a human decision (recorded with who/when), audit log; role-based API; Streamlit Alerts page panel |
| Trained artifacts in this ZIP | All six: diabetes, cardiovascular, hypertension (public mirrors), sepsis ML, causal Transformer/LSTM, forecasting, ECG (PhysioNet, trained on Colab). Each has provenance + a matching report; none is clinically validated |
