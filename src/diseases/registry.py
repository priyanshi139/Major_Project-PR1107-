"""Disease-specific architecture. Each disease keeps its own dataset, model, inputs and evaluation.
A model is only 'available' if it passed the availability gate (src/utils/provenance.py): trained artifact + recorded real-data provenance +
a matching evaluation report. 'pipeline_test' = a model object exists but is NOT validated (synthetic/unverified data or no evaluation).
The sepsis model is NEVER reused for other diseases.
To add a disease: train + evaluate a model on its own dataset, implement predict(history_df) -> dict(risk_score, threshold, ...),
register a DiseaseModule with status 'available', and give it its own namespace (e.g. 'diabetes:<id>')."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiseaseModule:
    key: str
    name: str
    status: str                       # 'available' | 'pipeline_test' | 'unavailable'
    dataset: str
    description: str
    patient_linked: bool = True       # False = works on its own records, not on registered patients' twins
    reason: str = ""
    model_info: dict = field(default_factory=dict)


def load_model_meta(artifacts_dir) -> dict:
    p = Path(artifacts_dir) / "sepsis" / "model_meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _gate_status(assets: dict, model_key: str, gate_key: str, missing_reason: str):
    """-> (status, reason, gate). Availability is decided ONLY by the gate, never by 'a model object exists'."""
    model, g = assets.get(model_key), (assets.get("gates") or {}).get(gate_key)
    if model is None:
        return "unavailable", (g or {}).get("reason") or missing_reason, g
    if g and g.get("passed"):
        return "available", "", g
    return "pipeline_test", (g or {}).get("reason") or "Model object supplied without verified provenance and evaluation; not validated.", g


def build_registry(assets: dict, artifacts_dir=None) -> dict:
    meta = load_model_meta(artifacts_dir) if artifacts_dir else {}
    m = assets.get("model")
    s_status, s_reason, s_gate = _gate_status(assets, "model", "sepsis", "Trained model not found. Run scripts/train_sepsis.py.")
    e_status, e_reason, _ = _gate_status(assets, "ecg_model", "ecg", "Run scripts/download_ecg.py and scripts/train_ecg.py.")
    prov = meta.get("provenance") or {}
    reg = {
        "sepsis": DiseaseModule("sepsis", "Sepsis", s_status, "PhysioNet/CinC Challenge 2019 (ICU, hourly)",
                                "Hourly sepsis risk from 40 base features + missing indicators + causal rolling features.", reason=s_reason,
                                model_info={"name": getattr(m, "name", None), "threshold": getattr(m, "threshold", None),
                                            "selected_by": meta.get("selection_rule"), "seed": meta.get("seed"),
                                            "patients_per_split": meta.get("split_patients"), "versions": meta.get("versions"),
                                            "data_origin": prov.get("data_origin"), "trained_at": meta.get("trained_at"), "validated": s_status == "available"}),
        "ecg": DiseaseModule("ecg", "Cardiac rhythm (ECG)", e_status, "MIT-BIH Arrhythmia Database (30-min recordings)",
                             "Beat-level classification of ECG records.", patient_linked=False, reason=e_reason),
    }
    from src.diseases.tabular import SPECS as TAB
    tab = assets.get("tabular") or {}
    for key, sp in TAB.items():
        st_, why, g = _gate_status({"m": tab.get(key), "gates": {key: (assets.get("gates") or {}).get(key)}}, "m", key,
                                   f"Not trained yet. Run scripts/download_tabular.py then scripts/train_tabular.py {key}.")
        tm = tab.get(key)
        reg[key] = DiseaseModule(key, sp.name, st_, sp.dataset, sp.task, patient_linked=True, reason=why,
                                 model_info={"name": getattr(tm, "name", None), "threshold": getattr(tm, "threshold", None),
                                             "data_origin": (g or {}).get("origin"), "trained_at": (g or {}).get("trained_at"),
                                             "validated": st_ == "available", "limitations": sp.limitations,
                                             "separate_from_sepsis": True})
    return reg
