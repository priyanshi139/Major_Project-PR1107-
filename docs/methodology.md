# Methodology

## 1. Task
Row-level (hourly) prediction of the dataset's `SepsisLabel` using only the current and earlier hours of one patient. This is a research prototype task; it is not the official challenge utility score.

## 2. Data and splits
- Patient, row and sepsis counts are printed by `prepare_sepsis_data.py` and stored in `data/processed/sepsis_splits.json` and the model metadata; they depend on how many files you download. (An earlier 6,000-patient run reported 7.45% septic patients and 1.84% positive rows; these numbers are not shipped and must be re-measured on your data.)
- Patient-wise split: 70% / 15% / 15% (4,200 / 900 / 900 patients), stratified by whether the patient is ever septic, seed 42. Each patient appears in exactly one split (checked in code and tests).
- Sets A and B come from different hospital systems; they are pooled here. A set-A-train / set-B-test external check is listed as future work.

## 3. Leakage control
- Split by patient, not by row.
- Forward-fill is applied within each patient using past values only.
- Medians, means and standard deviations are fit on training patients only.
- Rolling and difference features use current/past rows only.
- Thresholds are chosen on the validation set; the test set is used once for reporting.
- Model selection uses validation AUPRC, not test results.
- A unit test checks that changing future rows does not change the score at an earlier hour.

## 4. Models
- Logistic Regression (class_weight=balanced).
- HistGradientBoostingClassifier (class_weight=balanced).
Both trained on the same causal features. No sequence model (LSTM/Transformer) was trained: not justified within the time budget; listed as future work.

## 5. Evaluation
AUROC, AUPRC (with the random baseline = prevalence), precision, recall, F1 at the validation-chosen threshold, confusion matrix, Brier score and calibration bins. Accuracy is deliberately not reported as a headline metric: with 1.8% positives, predicting "no sepsis" always gives ~98% accuracy and is useless.
Threshold = maximum validation F1. Trade-off: lowering the threshold raises recall and lowers precision (more false alarms); raising it does the opposite. This choice is not medically approved.

## 6. Digital Twin
Per-subject object holding arrived observations, latest values with "hours since measured", a baseline (first 8 hours) and recent mean (last 6 hours), data-quality label, latest risk, deviations, status transitions and a snapshot per hour (`state_at(hour)`).

## 7. Anomaly detection
For each measured vital, compare the new reading with the patient's own last up-to-24 measured readings (at least 6 needed): z = (x - mean) / max(sd, 3% of mean). |z| >= 3 flags an "unusual reading". Past values only; missing values never flagged; a patient whose normal level is high is not flagged for being high. Flags are not diagnoses.

## 8. Alerts
`warning`: score >= model threshold. `critical`: score >= 0.90 (configurable; a prototype rule). `info`: anomaly. Cooldown 6 simulated hours per (subject, type, severity); escalation passes through. Alerts can be acknowledged. Alerts are model-generated warnings, not confirmed events, and never trigger treatment.

## 9. Replay
Historical rows are emitted in hour order, in a background thread, with start/pause/resume/reset and speed control. It simulates streaming; it is not a live feed. A labelled synthetic patient (`sepsis:SYNTHETIC_demo`) exists for demos when data are unavailable.

## 10. Reproducibility
Seed 42; package versions saved in `artifacts/sepsis/model_meta.json`; configuration in `config/config.yaml`; commands in README. Regenerate: `download_sepsis.py` -> `prepare_sepsis_data.py` -> `train_sepsis.py` -> `generate_eval_doc.py`.

## 11. Not implemented (stated plainly)
ECG module (MIT-BIH), trend forecasting, glucose module, SHAP explanations for the boosting model, sequence models.

## 12. Patient platform additions
- **Observation entry:** values are validated against plausibility ranges (typing-error checks, not clinical reference ranges). Blank = missing and stays missing in storage; the model's existing preprocessing handles it (carry-forward, training median, missing indicators). Age, gender, ICU unit and ICU-stay hour come from the profile / monitoring start; `SepsisLabel` is never an input.
- **Timestamps to hours:** the model is hourly. An observation's hour = whole hours since the monitoring start; entries in the same hour are placed in the next slot (the user is told). Timestamps must move forward.
- **Risk categories:** Low / Elevated / High / Critical are display bands relative to the validation-chosen model threshold (Elevated from 0.5x threshold) and a configurable critical level (default 0.90). They are not clinical categories.
- **Alert rules (configurable):** model score >= threshold (warning, or critical at the critical level); score rising >= 0.20 within 3 observations while below threshold (warning); >= 2 vitals unusual versus the patient's own baseline at the same time (warning, domain "vital signs"); one unusual vital (info). Duplicates merge for 6 h; unacknowledged critical alerts escalate after 3 h (review status only).
- **Explanation:** occlusion analysis - set one variable's measured values to the training median, keep which readings were taken, re-score with the real model, report the change. Works for the selected boosting model; it is not SHAP and not causal.
- **What-if (current state):** replace chosen vitals in the last N observations with hypothetical values and score with the same model using only data up to now. Stored separately as simulated. It is a model-sensitivity tool, not a treatment simulation. The trajectory what-if shifts vitals from an earlier hour onward and re-scores every observed hour.
- **Simulated live data:** scenario drifts + noise for 7 vitals (labs never generated), always labelled `simulated_live`.
