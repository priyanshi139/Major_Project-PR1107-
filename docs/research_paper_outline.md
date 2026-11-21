# Research Paper Outline (draft - fill with verified results only)

**Working title:** HealthTwin: A Software-Only Digital Twin Framework for Replay-Based Sepsis Risk Monitoring and Alerting

1. **Abstract** - problem, system, data, measured results (from reports/sepsis_metrics.json), limitations. 150-200 words.
2. **Introduction** - sepsis and early warning; digital twins in healthcare; why a software-only replay prototype; contributions stated modestly (a working, tested, leakage-aware architecture; honest evaluation).
3. **Related work** - sepsis prediction and the 2019 challenge; digital twins in health; alert fatigue; explainability. (See reference list below; verify every entry yourself.)
4. **Methodology** - dataset, patient-wise splits, leakage control, features, models, thresholding, Twin, anomaly detection, alerts, replay (docs/methodology.md).
5. **Experimental setup** - hardware (laptop, no GPU), software versions (model_meta.json), seed, split sizes.
6. **Results** - AUROC/AUPRC vs random baseline, precision/recall/F1, confusion matrix, calibration, example patient timelines (one detected, one missed), system behaviour (replay, alerts, suppressed duplicates).
7. **Discussion** - what the numbers mean; false-alarm burden; why scores are not probabilities; replay vs real deployment.
8. **Limitations** - docs/limitations_and_ethics.md.
9. **Conclusion and future work.**
10. **References.**

## Candidate references (from the developer's knowledge - VERIFY title, authors, year, pages and DOI before citing)
- Reyna MA, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine, 2020.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 2000.
- Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA, 2016.
- Corral-Acero J, et al. The 'Digital Twin' to enable the vision of precision cardiology. European Heart Journal, 2020.
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 2001. (for the future ECG module)

## Research gap
Do not claim novelty yet. First: (a) search Google Scholar / PubMed for "digital twin" + "sepsis" + "replay" + "alert"; (b) list what existing systems already do; (c) only then state a gap. Possible angles to investigate, not claims: a fully reproducible, leakage-tested, open software architecture that couples a patient Twin with per-patient baseline anomaly alerts and cooldown logic. Experiments needed to substantiate any claim: comparison with simple threshold rules (e.g. a basic early-warning score), alert burden per patient-day, and external validation.
