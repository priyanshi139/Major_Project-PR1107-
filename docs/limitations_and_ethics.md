# Limitations and Ethics

## Statement
HealthTwin is an academic prototype. It is not a medical device, is not clinically validated, and must not be used to diagnose, monitor or treat real patients. It never prescribes treatment.

## Technical limitations
- Sepsis models are only as good as the evaluation you generate locally: `scripts/train_sepsis.py` writes `reports/model_evaluation.json` and `.md`. Sepsis data are highly imbalanced, so expect many false alarms; read the report, not this document.
- All models ship trained on public data (tabular: GitHub mirrors; sepsis, forecasting, ECG: PhysioNet, trained on Colab); no raw dataset ships. A model shows as available only if it was trained on real public data (recorded in `PROVENANCE.json`) and has a matching evaluation report; models trained on synthetic or unverified data are labelled PIPELINE TEST and are not validated.
- Training on a subset (`--limit`) reduces quality; the subset size is recorded in the model metadata.
- Scores come from class-weighted models and are ranking scores, not calibrated probabilities.
- Single dataset, no external validation, no prospective evaluation.
- Row-level evaluation; patient-level early-warning metrics and the official challenge utility score were not computed.
- Explanations use occlusion (replace a feature by its baseline and see the score change); SHAP is not implemented yet. They show association, not causation.
- Anomaly detection depends on a short per-patient history and fixed thresholds; sparse measurements reduce reliability.
- Replay speed and timing are simulated; real devices would add noise, delays, outages and artefacts not modelled here.
- Forecasting and ECG modules exist as code but have no results until trained on real data. Glucose/CGM is not implemented.

## Data ethics
- Public, de-identified research data (PhysioNet). Follow the license/terms on the dataset page and cite the dataset.
- No attempt is made to re-identify patients. Raw data and trained artifacts are not committed to the repository (.gitignore).
- Datasets are namespaced; unrelated datasets are never treated as the same person.
- Synthetic data are always labelled `SYNTHETIC` and are never presented as real.

## Safety design
- Visible disclaimer on every dashboard page; simulation banner on the overview.
- Alerts say "model-generated warning / unusual reading", not "diagnosis".
- Human oversight is assumed: acknowledgement is manual; no automated action.
- Thresholds are configurable and documented as prototype rules, not medically approved.

## Bias and fairness (not assessed)
Performance across age, gender, ICU unit and hospital set was not analysed. Differences between set A and set B hospitals could affect generalisation. A subgroup analysis is future work.

## Patient platform notes
- Patients are marked SYNTHETIC by default; do not enter real names or identifiers. Only the profile fields shown in the app are stored (in a local SQLite file).
- Manual entries are exactly what is typed; plausibility checks catch typing errors but do not validate clinical correctness.
- Diabetes, cardiovascular and hypertension models can be run as patient-linked point-in-time assessments and stored in the patient twin history when validated artifacts are available. They are not temporal forecasts and do not automatically generate alerts.
- Tabular datasets are small and old or single-site: Pima (768 women of one ancestry), Framingham teaching subset (4,240 people, mid-20th-century US cohort), Performance is modest (cardiovascular); no external validation.
- The Framingham and Pima files used by the download script are public GitHub mirrors, not the original distributions; their size and prevalence are checked against the published values and any mismatch is written into the report.
- Hypertension is a cross-sectional screening estimate (blood-pressure readings are excluded to avoid label leakage); a real BP measurement is the proper test.
- What-if results show how the model responds to changed inputs; they are not treatment effects, recommendations or predictions of outcomes.
- Risk categories and alert thresholds are configurable display/decision-support rules, not clinically validated cut-offs.
- The occlusion explanation describes the model, not physiology or causation.
- The Streamlit app is a local single-user prototype (no login). The FastAPI backend has JWT login, roles and an audit log, but no audit signing, password reset, MFA or concurrency testing; do not expose it to a network with real patient data.
- The triage agent ranks and proposes only; it is deterministic, uses model outputs that are not clinically validated, and every proposal needs a clinician's decision.
