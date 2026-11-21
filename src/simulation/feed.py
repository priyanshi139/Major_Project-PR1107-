"""Simulated live data generator. ALWAYS labelled 'Simulated Live Data' / source 'simulated_live' or 'synthetic_demo'.
Simple scenario drifts + noise (not a physiological model); labs are never generated."""
from collections import namedtuple
import numpy as np
from src.digital_twin.whatif import LIMITS

FeedItem = namedtuple("FeedItem", ["hour", "values"])
SCENARIOS = {"stable": "Stable - vitals fluctuate around their current level",
             "deterioration": "Gradual deterioration - HR/Resp/Temp drift up, BP/O2Sat drift down",
             "recovery": "Recovery - vitals relax toward typical resting values"}
BASE = {"HR": 80.0, "Resp": 16.0, "Temp": 36.8, "O2Sat": 97.0, "SBP": 120.0, "MAP": 85.0, "DBP": 70.0}
DRIFT = {"HR": 2.5, "Resp": 0.6, "Temp": 0.12, "O2Sat": -0.4, "SBP": -1.8, "MAP": -1.4, "DBP": -1.0}
NOISE = {"HR": 2.0, "Resp": 0.7, "Temp": 0.08, "O2Sat": 0.5, "SBP": 3.0, "MAP": 2.0, "DBP": 2.0}


def generate_feed(scenario="stable", n_hours=12, start=None, seed=0, onset=0, missing_rate=0.10) -> list:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'")
    rng = np.random.default_rng(seed)
    state = {k: float((start or {}).get(k, BASE[k])) for k in BASE}
    items = []
    for k in range(n_hours):
        if k >= onset:
            for v in BASE:
                if scenario == "deterioration":
                    state[v] += DRIFT[v]
                elif scenario == "recovery":
                    state[v] += (BASE[v] - state[v]) * 0.25
        vals = {}
        for v in BASE:
            lo, hi = LIMITS[v]
            x = float(np.clip(state[v] + rng.normal(0, NOISE[v]), lo, hi))
            vals[v] = None if rng.random() < missing_rate else round(x, 1)
        if all(vals[v] is None for v in BASE):                      # never emit an empty observation
            vals["HR"] = round(state["HR"], 1)
        items.append(FeedItem(k, vals))
    return items
