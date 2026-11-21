# Future Work
1. ECG module with MIT-BIH (record-wise split, beat-class metrics) in its own namespace.
2. Trend forecasting for selected vitals, compared with a persistence baseline (MAE/RMSE) and shown only where it beats the baseline.
3. Use all ~40,000 patients; tune models; try sequence models (LSTM/Transformer) if compute allows.
4. External validation (train on set A, test on set B); patient-level early-warning metrics and the official challenge utility score.
5. Calibration (e.g. isotonic/Platt on validation data) and SHAP for the boosting model.
6. Subgroup / fairness analysis.
7. Optional glucose module after verifying a suitable public CGM dataset.
8. Better anomaly methods (Isolation Forest, change-point detection) and comparison against the rolling baseline.
9. Authentication and audit logging if ever moved beyond a classroom demo.
