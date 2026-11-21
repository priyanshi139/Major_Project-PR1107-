"""What-if analysis = MODEL SENSITIVITY on observed history. NOT a treatment-effect or outcome simulation.
We shift selected measured vitals from a chosen hour onward and re-score the already-observed hours with the SAME model
(each hour using only data up to that hour). No future values are invented."""
import numpy as np
import pandas as pd

ADJUSTABLE = {"HR": ("Heart rate", "beats/min", 30.0), "Resp": ("Respiratory rate", "breaths/min", 15.0),
              "MAP": ("Mean arterial pressure", "mm Hg", 30.0), "SBP": ("Systolic BP", "mm Hg", 40.0),
              "O2Sat": ("Oxygen saturation", "%", 10.0), "Temp": ("Temperature", "deg C", 3.0)}
LIMITS = {"HR": (20, 250), "Resp": (4, 60), "MAP": (30, 160), "SBP": (50, 250), "O2Sat": (50, 100), "Temp": (30, 43),
          "DBP": (20, 150), "EtCO2": (5, 100)}


def apply_adjustments(history: pd.DataFrame, adjustments: dict, start_hour: int) -> pd.DataFrame:
    h = history.copy()
    for var, delta in adjustments.items():
        if var not in LIMITS:
            raise ValueError(f"'{var}' cannot be adjusted")
        if not delta:
            continue
        m = (h["hour"] >= start_hour) & h[var].notna()
        lo, hi = LIMITS[var]
        h.loc[m, var] = (h.loc[m, var] + delta).clip(lo, hi)
    return h


def risk_series(model, history: pd.DataFrame, first_hour: int) -> pd.DataFrame:
    rows = []
    for t in history["hour"][history["hour"] >= first_hour]:
        r = model.predict(history[history["hour"] <= t])
        rows.append({"hour": int(t), "risk_score": r["risk_score"], "flagged": bool(r["flagged"])})
    return pd.DataFrame(rows, columns=["hour", "risk_score", "flagged"])


def compare_scenarios(model, history: pd.DataFrame, adjustments: dict, start_hour: int, max_points=72) -> dict:
    if model is None:
        raise RuntimeError("No trained model loaded")
    if history is None or len(history) == 0:
        raise ValueError("No observed history yet")
    cur = int(history["hour"].iloc[-1])
    first = max(0, min(start_hour, cur) - 2, cur - max_points + 1)
    scen_hist = apply_adjustments(history, adjustments, start_hour)
    base, scen = risk_series(model, history, first), risk_series(model, scen_hist, first)
    active = {k: v for k, v in adjustments.items() if v}
    return {"baseline": base, "scenario": scen, "adjustments": active, "start_hour": start_hour, "current_hour": cur,
            "delta_last": float(scen["risk_score"].iloc[-1] - base["risk_score"].iloc[-1]),
            "max_abs_delta": float(np.max(np.abs(scen["risk_score"].values - base["risk_score"].values))),
            "threshold": model.threshold,
            "note": "Model sensitivity analysis on observed history. Not a prediction of what a treatment would do."}


def scenario_state(model, history: pd.DataFrame, changes: dict, last_k: int = 1) -> dict:
    """State-based what-if: 'what if the CURRENT vital values were different?'
    Replaces the chosen vitals in the last `last_k` observations with the hypothetical values and scores the current state
    with the SAME trained model (still only data up to now). Nothing is written to the patient's observations."""
    if history is None or len(history) == 0:
        raise ValueError("No observed history yet")
    clean = {}
    for var, val in changes.items():
        if var not in LIMITS:
            raise ValueError(f"'{var}' cannot be used in a scenario")
        if val is None:
            continue
        lo, hi = LIMITS[var]
        if not (lo <= float(val) <= hi):
            raise ValueError(f"{var}={val} is outside the plausible range {lo}-{hi}")
        clean[var] = float(val)
    k = max(1, min(int(last_k), len(history)))
    scen = history.copy(); idx = scen.index[-k:]
    rows = []
    for var in LIMITS:
        meas = history[var].dropna()
        cur = float(meas.iloc[-1]) if len(meas) else None                      # carried-forward current value
        sim = clean.get(var, cur)
        if var in clean:
            scen.loc[idx, var] = clean[var]
        rows.append({"variable": var, "current": cur, "simulated": sim, "change": (sim - cur) if (sim is not None and cur is not None) else None,
                     "changed": var in clean})
    out = {"table": pd.DataFrame(rows), "changes": clean, "applied_to_last": k, "hour": int(history["hour"].iloc[-1]),
           "current": None, "scenario": None, "delta_score": None, "threshold": getattr(model, "threshold", None),
           "note": "Hypothetical values scored by the trained model. Not a treatment effect and not medical advice. "
                   "Vitals are physiologically linked (e.g. MAP with SBP/DBP); only the vitals you changed are modified."}
    if model is None:
        out["reason"] = "No trained model is loaded, so no score was calculated; the scenario is shown as a value comparison only."
    elif not clean:
        out["reason"] = "No values were changed."
        out["current"] = model.predict(history)
    else:
        out["current"], out["scenario"] = model.predict(history), model.predict(scen)
        out["delta_score"] = out["scenario"]["risk_score"] - out["current"]["risk_score"]
    return out
