# Faculty Demo Script (about 8-10 minutes)

Before the demo: activate the venv, run `python -m pytest -q` once (all pass), then `streamlit run app.py`.

1. **Framing (30 s).** "HealthTwin is a software-only prototype. It replays historical ICU data to simulate monitoring. It is not a live system and not a medical device." Point at the disclaimer banner.
2. **Overview page.** Show subjects, model name, status. State the pipeline: data -> replay -> Digital Twin -> model -> anomaly -> alerts -> database.
3. **Pick a patient** (sidebar: "Only patients with SepsisLabel"). Press **Start**; raise speed to ~5.
4. **Patient / Twin view.** Latest vitals, "hours since measured", missing-data map, baseline vs recent. Explain that the Twin only knows hours that have already arrived.
5. **Risk monitoring.** Show score vs model threshold. Red squares = dataset label (not used by the model). Say what the score is NOT: not a diagnosis, not a probability.
6. **Pause / Resume / Step / Reset.** Show that the UI does not freeze and replay resumes correctly.
7. **Alerts page.** Show explanation text, severity, cooldown ("duplicates suppressed"), acknowledge one alert.
8. **Show both outcomes.** One patient where the model flags early and one where it misses. Say clearly that recall is low.
9. **Model evaluation page.** AUROC/AUPRC vs the random baseline, confusion matrix, limitations. Explain why accuracy is not used (1.8% positives) and how the threshold was chosen on validation data.
10. **Honest scope (30 s).** ECG and forecasting pages say "not implemented"; those are future work.
11. **Engineering evidence.** Mention the automated tests (leakage test, replay pause/resume, alert cooldown, dashboard startup without artifacts).

Likely questions: "Why is precision low?" (rare positives, simple features, 15% of data size, no tuning); "Is it validated?" (no); "Why a Digital Twin and not just a model?" (the Twin holds state, baseline, history, alerts and transitions; the model is one component); "Does replay use future data?" (no; tested).
