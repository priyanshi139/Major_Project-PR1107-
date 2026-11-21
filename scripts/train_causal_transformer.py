"""Train and evaluate causal Transformer/LSTM sepsis sequence models.

Run after scripts/prepare_sepsis_data.py. Example:
  python scripts/train_causal_transformer.py --epochs 8 --batch-size 256
  python scripts/train_causal_transformer.py --max-patients 300 --epochs 2  # pipeline smoke run
"""
import argparse, json, platform, sys, importlib.metadata as md
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss
from src.preprocessing.sepsis_preprocessing import SepsisPreprocessor, check_split_integrity
from src.models.causal_transformer import CausalTransformer, SequenceLSTM, make_patient_windows
from src.models.sepsis_features import build_features
from src.evaluation.metrics import choose_threshold_max_f1
from src.utils.config import load_config, resolve_path
from src.utils.provenance import now_utc


def metrics(y, p, threshold):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required in the evaluation split")
    pred = (p >= threshold).astype(int)
    return {"n_windows": int(len(y)), "n_positive_windows": int(y.sum()),
            "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "random_auprc_baseline": float(y.mean()),
            "threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "brier": float(brier_score_loss(y, p))}


def train_one(model, loaders, device, epochs, lr, pos_weight):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_state, best_loss = None, float("inf")
    history = []
    for epoch in range(epochs):
        model.train(); tr_loss = 0.0; tr_n = 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += float(loss.item()) * len(yb); tr_n += len(yb)
        model.eval(); va_loss = 0.0; va_n = 0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                va_loss += float(loss.item()) * len(yb); va_n += len(yb)
        va_loss /= max(va_n, 1)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / max(tr_n, 1), "val_loss": va_loss})
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model, loader, device):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().tolist())
    return np.asarray(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-patients", type=int, default=None, help="Optional patient cap for smoke runs; such results are not full-cohort results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = ap.parse_args()
    if a.epochs < 1 or a.seq_len < 2 or a.stride < 1:
        ap.error("epochs>=1, seq-len>=2 and stride>=1 are required")
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device == "cuda" and not torch.cuda.is_available(): ap.error("CUDA requested but unavailable")
    device = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    cfg = load_config(); proc = resolve_path(cfg, "processed_dir")
    pq, sj = proc / "sepsis_raw_combined.parquet", proc / "sepsis_splits.json"
    if not pq.exists() or not sj.exists():
        raise SystemExit("Processed sepsis data not found. Run python scripts/prepare_sepsis_data.py first.")
    df = pd.read_parquet(pq); split_doc = json.loads(sj.read_text(encoding="utf-8"))
    splits = split_doc["splits"]; check_split_integrity(splits)
    if a.max_patients:
        # Deterministic cap independently within each split; keeps patient-wise separation.
        splits = {k: sorted(v)[:a.max_patients] for k, v in splits.items()}
    parts = {k: df[df.subject_id.isin(v)].copy() for k, v in splits.items()}
    if any(v.empty for v in parts.values()): raise SystemExit("One or more patient splits are empty after applying the cap.")
    pre = SepsisPreprocessor().fit(parts["train"])
    transformed = {k: pre.transform(v) for k, v in parts.items()}
    feat_cols = [c for c in transformed["train"].columns if c not in ("subject_id", "hour", "SepsisLabel")]
    arrays = {k: make_patient_windows(v, feat_cols, seq_len=a.seq_len, stride=a.stride) for k, v in transformed.items()}
    ds = {k: TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) for k, (x, y, _, _) in arrays.items()}
    loaders = {k: DataLoader(v, batch_size=a.batch_size, shuffle=(k == "train"), num_workers=0) for k, v in ds.items()}
    npos = float(arrays["train"][1].sum()); nneg = float(len(arrays["train"][1]) - npos)
    if npos == 0 or nneg == 0: raise SystemExit("Training split must contain positive and negative hourly labels.")
    pw = min(nneg / npos, 100.0)
    models = {"causal_transformer": CausalTransformer(len(feat_cols), max_len=a.seq_len),
              "lstm": SequenceLSTM(len(feat_cols))}
    results, histories, fitted = {}, {}, {}
    for name, model in models.items():
        model, hist = train_one(model, loaders, device, a.epochs, 1e-3, pw)
        histories[name] = hist; fitted[name] = model
        vp = predict(model, loaders["val"], device)
        threshold = choose_threshold_max_f1(arrays["val"][1], vp)
        tp = predict(model, loaders["test"], device)
        results[name] = {"validation": metrics(arrays["val"][1], vp, threshold),
                         "test": metrics(arrays["test"][1], tp, threshold),
                         "threshold_source": "maximum F1 on validation split only"}
    selected = max(results, key=lambda n: results[n]["validation"]["auprc"])
    # Save transformer separately regardless of whether it wins; selected model metadata is explicit.
    art = resolve_path(cfg, "artifacts_dir") / "sepsis_transformer"; repdir = resolve_path(cfg, "reports_dir")
    art.mkdir(parents=True, exist_ok=True); repdir.mkdir(parents=True, exist_ok=True)
    transformer = fitted["causal_transformer"]
    torch.save({"state_dict": transformer.state_dict(), "n_features": len(feat_cols),
                "feature_names": feat_cols, "seq_len": a.seq_len, "d_model": 64, "n_heads": 4,
                "n_layers": 2, "dropout": 0.1}, art / "causal_transformer.pt")
    joblib.dump(pre, art / "preprocessor.joblib")
    trained_at = now_utc(); provenance = split_doc.get("provenance") or {"data_origin": "unverified", "source": "missing provenance in split manifest"}
    meta = {"trained_at": trained_at, "model_type": "causal_transformer", "selected_sequence_model": selected,
            "selection_rule": "highest validation AUPRC among causal Transformer and LSTM",
            "seed": a.seed, "seq_len": a.seq_len, "stride": a.stride, "epochs": a.epochs,
            "device": str(device), "feature_names": feat_cols, "split_patients": {k: len(v) for k, v in splits.items()},
            "split_windows": {k: int(len(arrays[k][1])) for k in arrays},
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": md.version("numpy"), "pandas": md.version("pandas")},
            "provenance": provenance, "smoke_subset": bool(a.max_patients),
            "disclaimer": "Research prototype; not clinically validated and not for diagnosis or treatment."}
    (art / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Optional comparison to the already-trained row-wise V4 baseline, on exactly
    # the same sampled hourly endpoints. It is reported only if its artifact exists.
    baseline_comparison = {"status": "not_available", "reason": "No existing artifacts/sepsis/model.joblib baseline artifact."}
    baseline_path = resolve_path(cfg, "artifacts_dir") / "sepsis" / "model.joblib"
    if baseline_path.exists():
        try:
            baseline_obj = joblib.load(baseline_path)
            baseline_model = baseline_obj.get("model") if isinstance(baseline_obj, dict) else baseline_obj
            baseline_results = {}
            for split_name, frame in transformed.items():
                feat = build_features(frame)
                # make_patient_windows emits endpoint order grouped by patient/hour.
                _, y_end, pid_end, hour_end = arrays[split_name]
                keys = pd.DataFrame({"subject_id": pid_end, "hour": hour_end})
                keys["_order"] = np.arange(len(keys))
                joined = keys.merge(frame[["subject_id", "hour"]].assign(_row=np.arange(len(frame))),
                                    on=["subject_id", "hour"], how="left", validate="one_to_one")
                if joined["_row"].isna().any():
                    raise ValueError("Could not align sequence endpoints to baseline rows")
                Xb = feat.iloc[joined["_row"].astype(int).to_numpy()]
                if isinstance(baseline_obj, dict) and baseline_obj.get("feature_names"):
                    Xb = Xb.reindex(columns=baseline_obj["feature_names"], fill_value=0.0)
                pb = baseline_model.predict_proba(Xb)[:, 1]
                baseline_results[split_name] = metrics(y_end, pb, float(baseline_obj.get("threshold", 0.5)) if isinstance(baseline_obj, dict) else 0.5)
            baseline_comparison = {"status": "evaluated", "results": baseline_results,
                                   "note": "Existing row-wise V4 baseline evaluated on the same sampled hourly endpoints; threshold is its stored training threshold."}
        except Exception as exc:
            baseline_comparison = {"status": "not_evaluated", "reason": f"Baseline artifact found but comparison failed: {exc}"}
    report = {"meta": meta, "results": results, "training_history": histories,
              "rowwise_baseline_comparison": baseline_comparison,
              "comparison_note": "Transformer and LSTM use identical patient-wise splits and sequence windows. Metrics are hourly-window metrics, not patient-level clinical utility."}
    (repdir / "sepsis_transformer_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "results": results, "artifact_dir": str(art), "report": str(repdir / 'sepsis_transformer_metrics.json')}, indent=2))

if __name__ == "__main__": main()
