"""Exact model input schema (same names/order as the PhysioNet-2019 pipeline) + plausibility ranges for data entry.
Ranges are PLAUSIBILITY limits to catch typing errors, not clinical reference ranges."""
from dataclasses import dataclass
from src.data_ingestion.sepsis_loader import VITALS, LABS, FEATURES, DEMOGRAPHICS


@dataclass(frozen=True)
class FeatureSpec:
    name: str; label: str; unit: str; group: str; lo: float; hi: float


_V = [("HR", "Heart rate", "beats/min", 20, 250), ("O2Sat", "Oxygen saturation", "%", 50, 100), ("Temp", "Temperature", "deg C", 30, 43),
      ("SBP", "Systolic BP", "mm Hg", 50, 250), ("MAP", "Mean arterial pressure", "mm Hg", 30, 160), ("DBP", "Diastolic BP", "mm Hg", 20, 150),
      ("Resp", "Respiratory rate", "breaths/min", 4, 60), ("EtCO2", "End-tidal CO2", "mm Hg", 5, 100)]
_L = [("BaseExcess", "Base excess", "mmol/L", -30, 30), ("HCO3", "Bicarbonate", "mmol/L", 2, 60), ("FiO2", "Fraction of inspired O2", "fraction", 0.21, 1.0),
      ("pH", "Arterial pH", "", 6.5, 8.0), ("PaCO2", "Arterial CO2 pressure", "mm Hg", 5, 150), ("SaO2", "Arterial O2 saturation", "%", 30, 100),
      ("AST", "AST", "IU/L", 0, 20000), ("BUN", "Blood urea nitrogen", "mg/dL", 0, 300), ("Alkalinephos", "Alkaline phosphatase", "IU/L", 0, 3000),
      ("Calcium", "Calcium", "mg/dL", 2, 20), ("Chloride", "Chloride", "mmol/L", 60, 150), ("Creatinine", "Creatinine", "mg/dL", 0.1, 25),
      ("Bilirubin_direct", "Direct bilirubin", "mg/dL", 0, 50), ("Glucose", "Glucose", "mg/dL", 10, 1500), ("Lactate", "Lactate", "mg/dL", 0, 30),
      ("Magnesium", "Magnesium", "mmol/dL", 0.2, 10), ("Phosphate", "Phosphate", "mg/dL", 0.2, 20), ("Potassium", "Potassium", "mmol/L", 1, 10),
      ("Bilirubin_total", "Total bilirubin", "mg/dL", 0, 60), ("TroponinI", "Troponin I", "ng/mL", 0, 500), ("Hct", "Hematocrit", "%", 5, 70),
      ("Hgb", "Hemoglobin", "g/dL", 2, 25), ("PTT", "Partial thromboplastin time", "seconds", 10, 250), ("WBC", "White blood cell count", "x10^3/uL", 0, 200),
      ("Fibrinogen", "Fibrinogen", "mg/dL", 20, 1500), ("Platelets", "Platelet count", "x10^3/uL", 1, 1500)]
SPECS = {n: FeatureSpec(n, l, u, "Vital signs", lo, hi) for n, l, u, lo, hi in _V}
SPECS.update({n: FeatureSpec(n, l, u, "Laboratory", lo, hi) for n, l, u, lo, hi in _L})
assert list(SPECS) == VITALS + LABS, "schema must match the model's feature order"


def validate_values(values: dict):
    """-> (clean, errors). Blank/None = not measured (kept missing; never filled here). Unknown names are errors."""
    clean, errors = {}, []
    for k, v in values.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if k not in SPECS:
            errors.append(f"'{k}' is not a measurable model input"); continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errors.append(f"{SPECS[k].label}: '{v}' is not a number"); continue
        if x != x or not (SPECS[k].lo <= x <= SPECS[k].hi):
            errors.append(f"{SPECS[k].label} {x:g} {SPECS[k].unit} is outside the plausible range {SPECS[k].lo:g}-{SPECS[k].hi:g}"); continue
        clean[k] = x
    return clean, errors
