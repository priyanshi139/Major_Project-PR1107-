# HealthTwin 
AI-powered multi-disease Digital Twin with simulated replay monitoring. Not a medical device; not clinically validated.

## Setup (Windows PowerShell, from project root)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/check_environment.py
    python -m pytest -q

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Status (final: phases 1-7)
- Working: patient platform (11 pages), twin, alerts, what-if, occlusion insights, tests.
- Shipped TRAINED and evaluated on real public data (reports in `reports/`): diabetes (Pima), cardiovascular and hypertension (Framingham teaching subset), sepsis ML + causal Transformer/LSTM (PhysioNet/CinC 2019, 6,000 patients), vital-sign forecasting, ECG beat classifier (MIT-BIH, 48 records). The gate shows all six as VALIDATED; performance is modest (see reports) and not clinically validated.
- Training of the sepsis, sequence, forecasting and ECG models was done on Google Colab with `colab_train_remaining.ipynb` (your laptop can use `scripts/train_all.ps1` instead). Re-run it to reproduce.
- Phase 2: separate diabetes / cardiovascular / hypertension pipelines.
- Phase 3: validated tabular assessments are stored in the selected patient's Digital Twin history. They remain cross-sectional estimates, not temporal forecasts.
- Phase 4: local permutation-SHAP explanations for validated diabetes, cardiovascular and hypertension estimates.
- Phase 5: causal Transformer and LSTM sepsis sequence-training pipeline with patient-wise splits, causal masking, validation selection and held-out metrics. Trained on the real 6,000-patient subset (see `reports/sepsis_transformer_metrics.json`).
- Phase 7: agentic triage (Alerts page) that ranks patients and PROPOSES follow-ups; a clinician must approve or reject - nothing executes. FastAPI backend with JWT login and roles (patient / doctor / admin).
- Not implemented: login inside the Streamlit app (the API has it), temporal Transformer models for non-sepsis diseases, external validation, any clinical validation.

## Phase 7: API, login and agent (PowerShell)
    python scripts/create_user.py --username drmehta --role doctor --password "a-long-password"   # clinician accounts are not self-service
    uvicorn src.api.asgi:app --port 8000                                                          # interactive docs: http://127.0.0.1:8000/docs
    streamlit run app.py                                                                           # in a second window
Patients may self-register through `POST /api/auth/register`; doctor self-registration works only if you set `$env:HEALTHTWIN_DOCTOR_CODE`.
The agent is offline and deterministic (an optional LLM may only reword text and is discarded if it adds numbers; off by default).

## Train the remaining models without a local setup (Google Colab)
Open `colab_train_remaining.ipynb` in Colab, upload this ZIP, run the cells, and download `trained_results.zip` (sepsis ML, Transformer/LSTM, forecasting, ECG). Merge it into the project folder (artifacts, reports, data/processed), then run `python scripts/check_environment.py`.

## Model availability gate
A model is shown as AVAILABLE only if its artifact loads, it records real-data provenance and training time, and a matching evaluation report exists.
Models trained on synthetic or unverified data show as PIPELINE TEST and are not loaded unless `HEALTHTWIN_ALLOW_TEST_MODELS=1` (developers only).

## Other diseases (Phase 2): diabetes, cardiovascular and hypertension
Each disease has its OWN dataset, features, model, calibration, threshold, evaluation report and artifact folder; the sepsis model is never reused.
    python scripts/download_tabular.py              # Pima diabetes + Framingham teaching subset (public GitHub mirrors) + provenance
    python scripts/train_tabular.py                 # trains every disease whose data file exists; prints test AUROC, precision, recall, F1
    python scripts/train_tabular.py diabetes        # or one disease
Reports: `reports/<disease>_metrics.json` (+ curves png).
Diabetes = Pima (768 women), cardiovascular = 10-year CHD and hypertension = prevalent-hypertension screening (BP readings excluded) from the Framingham teaching subset.
Use them on the Prediction page. Select a patient to save validated assessments to that patient's Digital Twin history; without a selected patient, the estimate remains stand-alone.

## Quick real-data path for sepsis (PowerShell)
    python scripts/download_sepsis.py --limit 3000 --workers 8   # real PhysioNet files; records provenance
    python scripts/prepare_sepsis_data.py
    python scripts/train_sepsis.py
    python scripts/check_environment.py                          # shows data origin + gate state
    streamlit run app.py
Already downloaded the files yourself from PhysioNet? Run `python scripts/prepare_sepsis_data.py --confirm-physionet-origin` (or `--zip file.zip`), then `python scripts/train_sepsis.py`.
Models trained before Phase 1 have no provenance: re-run the two commands above to make them verifiable.

## Phase 2: sepsis data
1. Open https://physionet.org/content/challenge-2019/1.0.0/ , check the license/terms, and download the files (training_setA and training_setB, or the zip) manually.
2. Put them under data/raw/physionet_challenge_2019/ (or pass --zip).
3. Run:
       python scripts/prepare_sepsis_data.py
       python scripts/prepare_sepsis_data.py --limit 500   # quick trial
Outputs: data/processed/sepsis_raw_combined.parquet, data/processed/sepsis_splits.json
Tests use SYNTHETIC data only: python -m pytest -q

## Training, replay and dashboard
    python scripts/train_sepsis.py          # trains baselines, saves artifacts/sepsis + reports/
    python scripts/run_replay_cli.py        # terminal replay (add --synthetic for labelled synthetic demo)
    streamlit run app.py                    # interactive dashboard
Replay is a SIMULATION over historical data, not a live patient feed.

## Forecasting and ECG (optional)
    python scripts/train_forecast.py        # short-horizon vital forecasts vs persistence baseline
    python scripts/download_ecg.py          # MIT-BIH via wfdb (public PhysioNet data)
    python scripts/train_ecg.py             # record-wise beat classifier (DS1 train / DS2 test)
    streamlit run app.py
What-if is a model sensitivity analysis, not a treatment simulation. Forecast and ECG results appear only after you train them on real data.

## Patient-centric platform (current app)
`streamlit run app.py` - 11 pages: Overview, Patients, Profile, Twin, Live, Prediction, Timeline, What-if, Insights, Alerts, Data.
PowerShell: `.\.venv\Scripts\Activate.ps1` · CMD: `.venv\Scripts\activate.bat`. See `docs/platform_guide.md`, `docs/system_architecture.md`, `docs/PROJECT_STATUS.md`.
Patients and their history are stored in `artifacts/patients.db` (SQLite). Sepsis and the three tabular diseases have separate pipelines (each shows as available only after local training + evaluation); ECG is a separate record-level module.

## Phase 4 — local SHAP explanations

For trained and validated diabetes, cardiovascular and hypertension models, the Prediction page can display a local permutation-SHAP explanation after an estimate. The explanation uses the saved training background and the calibrated model score. It is associational, not causal or clinically validated. See `docs/PHASE4_REPORT.md` for scope and limitations.

## Phase 5: causal Transformer (sepsis hourly sequences)

After the real PhysioNet/CinC data have been prepared and patient-wise splits saved, run:

    python scripts/train_causal_transformer.py --epochs 8 --batch-size 256

Optional quick software smoke run (not a research result):

    python scripts/train_causal_transformer.py --max-patients 300 --epochs 2

The script trains a causal attention Transformer and an LSTM comparator on the same patient-wise splits and sequence windows, selects by validation AUPRC, chooses a threshold on validation only, and evaluates the held-out test split. Outputs: `artifacts/sepsis_transformer/` and `reports/sepsis_transformer_metrics.json`. CPU is supported; CUDA can be selected with `--device cuda`. See `docs/PHASE5_REPORT.md`. The model is sepsis-only because the other disease datasets are cross-sectional.


## Phase 6 Twin analytics

Patient snapshots now include `health_trajectory` (descriptive trends from at least three measured time points) and `baseline_quality`. These are patient-specific summaries derived from stored observations. Trend output is explicitly non-clinical and is not a validated forecast. See `docs/PHASE6_REPORT.md`.
