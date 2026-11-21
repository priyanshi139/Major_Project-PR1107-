"""Short-horizon forecasting of a vital sign, compared against a persistence baseline.
Features at row t use only rows <= t (last value carried forward, lags, hours since last measurement)."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORECAST_VARS = ["HR", "Resp", "SBP", "MAP", "O2Sat", "Temp"]
HORIZONS = [1, 3]
LAGS = 6
MIN_IMPROVEMENT = 0.02          # model must beat persistence by >= 2% MAE on VALIDATION to be enabled
DISCLAIMER = "Statistical short-term forecast of a measurement; not a prediction of disease or outcome."


def feature_frame(df: pd.DataFrame, var: str) -> pd.DataFrame:
    d = df.sort_values(["subject_id", "hour"])
    sid = d["subject_id"]
    ff = d.groupby("subject_id", sort=False)[var].ffill()
    X = pd.DataFrame({f"lag{k}": ff.groupby(sid).shift(k) for k in range(LAGS)}, index=d.index)
    last_meas = d["hour"].where(d[var].notna()).groupby(sid).ffill()
    X["hours_since_measured"] = d["hour"] - last_meas
    if "ICULOS" in d.columns:
        X["ICULOS"] = d["ICULOS"].groupby(sid).ffill()
    return X


def make_supervised(df: pd.DataFrame, var: str, horizon: int):
    d = df.sort_values(["subject_id", "hour"])
    X = feature_frame(d, var)
    y = d.groupby("subject_id", sort=False)[var].shift(-horizon)      # label only; never a feature
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def _fit(X, y):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(X, y - X["lag0"])                                         # learn the change from last value
    return pipe


def _predict(pipe, X):
    return X["lag0"].values + pipe.predict(X)


def train_forecasters(df, splits, variables=FORECAST_VARS, horizons=HORIZONS, min_rows=200):
    part = {k: df[df.subject_id.isin(v)] for k, v in splits.items()}
    models, metrics = {}, {}
    for var in variables:
        if var not in df.columns:
            continue
        for h in horizons:
            key = f"{var}|{h}"
            Xtr, ytr = make_supervised(part["train"], var, h)
            Xva, yva = make_supervised(part["val"], var, h)
            Xte, yte = make_supervised(part["test"], var, h)
            if min(len(Xtr), len(Xva), len(Xte)) < min_rows // 4 or len(Xtr) < min_rows:
                metrics[key] = {"variable": var, "horizon": h, "enabled": False,
                                "note": "not enough measured targets to train/evaluate"}
                continue
            pipe = _fit(Xtr, ytr)
            pv, pt = _predict(pipe, Xva), _predict(pipe, Xte)
            mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
            rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
            val_model, val_pers = mae(yva, pv), mae(yva, Xva["lag0"])
            enabled = val_model <= (1 - MIN_IMPROVEMENT) * val_pers
            sigma = float(np.std(np.asarray(yva) - pv))
            metrics[key] = {"variable": var, "horizon": h, "enabled": bool(enabled),
                            "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
                            "val_mae_model": val_model, "val_mae_persistence": val_pers,
                            "test_mae_model": mae(yte, pt), "test_mae_persistence": mae(yte, Xte["lag0"]),
                            "test_rmse_model": rmse(yte, pt), "test_rmse_persistence": rmse(yte, Xte["lag0"]),
                            "interval_sigma_val": sigma,
                            "enable_rule": f"model val MAE <= {1 - MIN_IMPROVEMENT:.2f} x persistence val MAE"}
            models[key] = {"pipe": pipe, "sigma": sigma, "enabled": bool(enabled)}
    return models, metrics


def save_forecasters(models, metrics, artifacts_dir, reports_dir, provenance=None):
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, artifacts_dir / "forecasters.joblib")
    from src.utils.provenance import PIPELINE_VERSION, now_utc
    meta = {"trained_at": now_utc(), "pipeline_version": PIPELINE_VERSION, "disclaimer": DISCLAIMER,
            "provenance": provenance or {"data_origin": "unverified", "source": "caller did not state the data origin"}}
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (reports_dir / "forecast_metrics.json").write_text(json.dumps(
        {"meta": meta, "disclaimer": DISCLAIMER, "min_improvement": MIN_IMPROVEMENT, "results": metrics}, indent=1), encoding="utf-8")


class ForecastService:
    def __init__(self, artifacts_dir):
        p = Path(artifacts_dir) / "forecasters.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Forecast models not found: {p}. Run scripts/train_forecast.py")
        self.models = joblib.load(p)
        self.horizons = sorted({int(k.split("|")[1]) for k in self.models})

    def predict(self, history: pd.DataFrame, var: str, horizon: int) -> dict:
        m = self.models.get(f"{var}|{horizon}")
        base = {"variable": var, "horizon": horizon, "available": False, "disclaimer": DISCLAIMER}
        if m is None:
            return {**base, "reason": "no model for this variable/horizon"}
        if not m["enabled"]:
            return {**base, "reason": "model did not beat the persistence baseline on validation data; "
                                      "showing historical trend only"}
        if history is None or len(history) == 0:
            return {**base, "reason": "no history yet"}
        X = feature_frame(history, var).iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return {**base, "reason": "not enough measured history (need 6 hours with a measurement)"}
        v = float(_predict(m["pipe"], X)[0]); s = m["sigma"]
        return {**base, "available": True, "value": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s,
                "persistence": float(X["lag0"].iloc[0]), "target_hour": int(history["hour"].iloc[-1]) + horizon,
                "method": "Ridge on last 6 values (change from last value)",
                "uncertainty": "approx. 95% band from validation residual SD; assumes similar error everywhere"}
