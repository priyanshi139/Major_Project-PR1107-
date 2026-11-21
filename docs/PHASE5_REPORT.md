# HealthTwin V4 — Phase 5: Causal Transformer sequence modelling

## Implemented in this phase

- Added a PyTorch Transformer encoder with a strict upper-triangular causal attention mask. The model emits a score for the final hour in each supplied history window.
- Added a two-layer LSTM-style recurrent comparator (single recurrent layer in the current implementation) with the same window interface.
- Added patient-local, chronological fixed-length sequence construction. Windows never cross patient boundaries; short histories are left-padded by repeating their earliest available observation (rather than inventing an all-zero clinical measurement).
- Added a reproducible training/evaluation script using the existing patient-wise train/validation/test assignments. Preprocessing is fitted on training patients only.
- Selects a sequence model by validation AUPRC; chooses its classification threshold using validation F1 only; reports held-out test AUROC, AUPRC, precision, recall, F1 and Brier score.
- Records split sizes, provenance, versions, seed, model settings and whether a patient cap was used. A capped run is explicitly a smoke/subset run, not a full-cohort evaluation.
- Added tests proving future-token perturbations do not change earlier Transformer outputs and that windows remain patient-local.

## Data and clinical boundaries

This phase trains temporal sequence models for sepsis only, because the existing PhysioNet/CinC 2019 data has hourly patient observations. The diabetes, cardiovascular and hypertension datasets in this project are cross-sectional; no artificial time dimension is created for them.

The target remains the supplied hourly `SepsisLabel`. Results are hourly-window metrics, not automatically patient-level early-warning utility. The current implementation does not claim a validated clinical lead-time benefit. No model is available until a full real-data run, matching report, and the project's provenance gate all pass.

## Run

After preparing the real PhysioNet/CinC 2019 files and patient-wise splits:

```powershell
python scripts/train_causal_transformer.py --epochs 8 --batch-size 256
```

For a quick pipeline smoke test only (not research results):

```powershell
python scripts/train_causal_transformer.py --max-patients 300 --epochs 2
```

Artifacts are written under `artifacts/sepsis_transformer/`; metrics are written to `reports/sepsis_transformer_metrics.json`. CPU is supported. CUDA is used automatically when available, or can be selected with `--device cuda`.

## Verification

Run `python -m pytest -q tests/test_phase5_transformer.py`. The unit tests use small synthetic sequences to verify software invariants only; they are not model training or clinical performance evidence.
