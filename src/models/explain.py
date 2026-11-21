"""Local explanation for ANY sepsis model (incl. the boosting model, which has no coefficients).
Method = occlusion: for each measured variable, replace its measured values by the training-set median (a 'neutral' value),
keep which readings were taken, re-score with the real model, and report how much the current score changes.
This is NOT SHAP and not causal; it shows what the model's current score leans on for THIS patient."""
from src.data_ingestion.sepsis_loader import VITALS, LABS


def occlusion_explanation(model, history, top_k=6) -> dict:
    base = model.predict(history)["risk_score"]
    neutral = model.pre.medians_
    items = []
    for v in VITALS + LABS:
        col = history[v]
        if not col.notna().any():
            continue
        h2 = history.copy()
        h2.loc[h2[v].notna(), v] = float(neutral[v])
        effect = base - model.predict(h2)["risk_score"]
        items.append({"feature": v, "effect": float(effect), "current_value": float(col.dropna().iloc[-1]),
                      "neutral_value": float(neutral[v])})
    items.sort(key=lambda d: -abs(d["effect"]))
    hour = int(history["hour"].iloc[-1])
    return {"base_score": float(base), "items": items[:top_k], "hour": hour, "model_name": model.name,
            "method": "Occlusion: set one variable's measured values to the training median and re-score",
            "note": "Positive effect = the variable's actual values push this score up compared with a typical (median) value. "
                    "Association within the model, not causation. SHAP is not used."}
