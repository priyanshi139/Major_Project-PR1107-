# Data Dictionary - Sepsis module (PhysioNet/CinC Challenge 2019)

Source: https://physionet.org/content/challenge-2019/1.0.0/ (open access; read the license/terms on the page).
Files: one pipe-separated `.psv` per patient, one row per hour. Two sets: `training_setA` (20,336 patients) and `training_setB` (20,000 patients) on the server; the download script's default `--limit` takes the first N files of each set (e.g. 3,000 each = 6,000 patients); the actual count is recorded in the model metadata.
Units below follow the dataset page as recalled during development - **verify against the page before publishing**.

## Added by this project
| Column | Meaning |
|---|---|
| subject_id | `sepsis:<file stem>`, dataset-namespaced |
| hour | Row index within patient file (0,1,2,...) used as simulated time |
| source_set | A or B (from folder name) |

## Vital signs (8)
HR (beats/min), O2Sat (%), Temp (deg C), SBP, MAP, DBP (mm Hg), Resp (breaths/min), EtCO2 (mm Hg)

## Laboratory values (26)
BaseExcess (mmol/L), HCO3 (mmol/L), FiO2 (fraction), pH, PaCO2 (mm Hg), SaO2 (%), AST (IU/L), BUN (mg/dL), Alkalinephos (IU/L), Calcium (mg/dL), Chloride (mmol/L), Creatinine (mg/dL), Bilirubin_direct (mg/dL), Glucose (mg/dL), Lactate (mg/dL), Magnesium (mmol/dL), Phosphate (mg/dL), Potassium (mmol/L), Bilirubin_total (mg/dL), TroponinI (ng/mL), Hct (%), Hgb (g/dL), PTT (seconds), WBC (count x10^3/uL), Fibrinogen (mg/dL), Platelets (count x10^3/uL)

## Demographics / administrative (6)
Age (years), Gender (0 female, 1 male), Unit1 / Unit2 (ICU unit indicators), HospAdmTime (hours between hospital and ICU admission), ICULOS (ICU length of stay, hours)

## Label
`SepsisLabel` (0/1, hourly). As documented on the dataset page, labels in the training data are shifted so that a septic patient's label becomes 1 six hours before the clinical sepsis-onset time used by the challenge. A patient is called "septic" in this project if any row has label 1 (used only to stratify the split and to describe prevalence).

## Missingness
Most labs are measured rarely; vitals are measured more often. Missing values are NaN in the raw data.

## Derived features (models/sepsis_features.py)
For each vital: rolling mean (6 h), rolling std (6 h), 1-hour difference - all computed from the current and past rows only. Plus `<feature>_missing` indicators and the preprocessed (forward-filled, train-median-imputed, train-standardised) base features.

## Project data files
| File | Created by | Content |
|---|---|---|
| data/raw/physionet_challenge_2019/... | scripts/download_sepsis.py | Raw .psv files |
| data/processed/sepsis_raw_combined.parquet | scripts/prepare_sepsis_data.py | All loaded rows |
| data/processed/sepsis_splits.json | scripts/prepare_sepsis_data.py | Patient IDs per split, seed |
| artifacts/sepsis/{preprocessor,model}.joblib, model_meta.json | scripts/train_sepsis.py | Fitted pipeline, selected model, threshold, versions |
| reports/sepsis_metrics.json, sepsis_curves.png | scripts/train_sepsis.py | Metrics and curves |
| artifacts/healthtwin.db | replay runs | SQLite: observations, states, risks, alerts |
