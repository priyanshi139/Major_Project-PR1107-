import math, time
import numpy as np
import pandas as pd
from src.digital_twin.models import Observation, PatientState


def _isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class DigitalTwin:
    """Per-subject software state: observations, latest state, baseline, risk, deviations, status transitions.
    Only ever sees observations that have 'arrived' so far (no future data)."""

    def __init__(self, subject_id, module, variables, vitals, baseline_hours=8, recent_window=6):
        self.subject_id, self.module = subject_id, module
        self.variables, self.vitals = list(variables), list(vitals)
        self.baseline_hours, self.recent_window = baseline_hours, recent_window
        self.reset()

    def reset(self):
        self._obs, self._latest, self._last_measured = [], {}, {}
        self._snapshots, self.risk_history, self.transitions = [], [], []
        self.disease_risk_history = {}
        self.latest_risk, self.deviations = None, []
        self.status, self._last_update = "no_data", None

    # ---- updates ----
    def add_observation(self, obs: Observation):
        if obs.subject_id != self.subject_id or obs.module != self.module:
            raise ValueError(f"Observation {obs.subject_id}/{obs.module} does not belong to twin "
                             f"{self.subject_id}/{self.module}")
        if self._obs and obs.hour <= self._obs[-1].hour:
            raise ValueError(f"Out-of-order or duplicate observation: hour {obs.hour} after {self._obs[-1].hour}")
        self._obs.append(obs)
        for k, v in obs.values.items():
            if _isnum(v):
                self._latest[k] = float(v)
                self._last_measured[k] = obs.hour
        self._last_update = time.time()
        self.deviations, self.latest_risk = [], None
        self._refresh_status()
        self._snapshot()

    def record_deviations(self, deviations: list):
        self.deviations = list(deviations)
        self._refresh_status(); self._snapshot_replace()

    def record_risk(self, risk: dict):
        """risk = output of a disease-specific model. Stores which observations were used."""
        self.latest_risk = risk
        self.risk_history.append({"hour": self.current_hour, **{k: risk.get(k) for k in
                                  ("risk_score", "threshold", "flagged", "model_name",
                                   "n_observations_used", "last_hour_used")}})
        self._refresh_status(); self._snapshot_replace()

    def record_disease_risk(self, disease: str, risk: dict, hour=None):
        """Store a separately trained tabular disease assessment against this patient's twin timeline."""
        item = {"hour": self.current_hour if hour is None else hour, "disease": disease, **risk}
        self.disease_risk_history.setdefault(disease, []).append(item)
        self._snapshot_replace()
        return item

    def _refresh_status(self):
        if not self._obs:
            new = "no_data"
        elif self.latest_risk and self.latest_risk.get("flagged"):
            new = "high_risk_flag"
        elif self.deviations:
            new = "deviation_from_baseline"
        else:
            new = "stable"
        if new != self.status:
            self.transitions.append({"hour": self.current_hour, "from": self.status, "to": new})
            self.status = new

    # ---- retrieval ----
    @property
    def current_hour(self):
        return self._obs[-1].hour if self._obs else None

    @property
    def snapshots(self):
        return list(self._snapshots)

    @property
    def n_observations(self):
        return len(self._obs)

    def observation_records(self):
        return [{"hour": o.hour, "timestamp": o.timestamp, "source": o.source} for o in self._obs]

    def past_measured(self, var, window, before_hour):
        vals = [float(o.values[var]) for o in self._obs
                if o.hour < before_hour and _isnum(o.values.get(var))]
        return vals[-window:]

    def history_df(self, upto_hour=None) -> pd.DataFrame:
        rows = [{"subject_id": self.subject_id, "hour": o.hour,
                 **{v: o.values.get(v, np.nan) if _isnum(o.values.get(v)) else np.nan for v in self.variables}}
                for o in self._obs if upto_hour is None or o.hour <= upto_hour]
        return pd.DataFrame(rows, columns=["subject_id", "hour"] + self.variables)

    def baseline(self):
        out = {}
        first = self._obs[: self.baseline_hours]
        recent = self._obs[-self.recent_window:]
        for v in self.vitals:
            b = [float(o.values[v]) for o in first if _isnum(o.values.get(v))]
            r = [float(o.values[v]) for o in recent if _isnum(o.values.get(v))]
            if len(b) >= 3:
                out[v] = {"baseline_mean": float(np.mean(b)), "baseline_sd": float(np.std(b, ddof=1)),
                          "n_baseline": len(b), "recent_mean": float(np.mean(r)) if r else None}
        return out

    def data_quality(self):
        if not self._obs:
            return "no_data"
        last = self._obs[-1]
        frac = np.mean([_isnum(last.values.get(v)) for v in self.vitals]) if self.vitals else 1.0
        return "good" if frac >= 0.6 else ("partial" if frac >= 0.3 else "poor")

    def health_trajectory(self, window=6, min_points=3):
        """Describe recent measured-vital trends; this is not a clinical forecast.

        Slopes use only observations already received, ignore missing measurements,
        and are returned only when enough distinct time points are available.
        """
        if not self._obs:
            return {"available": False, "reason": "No observations are available.",
                    "method": "descriptive_linear_trend", "clinical_forecast": False}
        out = {}
        for variable in self.vitals:
            points = [(o.hour, float(o.values[variable])) for o in self._obs
                      if _isnum(o.values.get(variable))]
            points = points[-max(int(window), min_points):]
            if len(points) < min_points or len({h for h, _ in points}) < min_points:
                continue
            hours = np.asarray([h for h, _ in points], dtype=float)
            values = np.asarray([v for _, v in points], dtype=float)
            slope = float(np.polyfit(hours - hours[-1], values, 1)[0])
            latest = float(values[-1])
            # A tiny tolerance avoids labelling numerical noise as a meaningful direction.
            scale = max(abs(latest), 1.0)
            tolerance = 0.005 * scale
            direction = "increasing" if slope > tolerance else ("decreasing" if slope < -tolerance else "approximately_stable")
            out[variable] = {
                "latest_value": latest,
                "slope_per_hour": slope,
                "direction": direction,
                "n_points": len(points),
                "first_hour": int(hours[0]),
                "last_hour": int(hours[-1]),
                "data_source": "observed_history",
            }
        return {
            "available": bool(out),
            "reason": None if out else f"At least {min_points} measured time points per vital are required.",
            "method": "descriptive_linear_trend",
            "clinical_forecast": False,
            "window_observations": int(window),
            "variables": out,
        }

    def state(self) -> PatientState:
        h = self.current_hour
        return PatientState(
            subject_id=self.subject_id, module=self.module, hour=h if h is not None else -1,
            n_observations=self.n_observations, last_update_wall_time=self._last_update or 0.0,
            status=self.status, data_quality=self.data_quality(), latest_values=dict(self._latest),
            hours_since_measured={k: (h - t) for k, t in self._last_measured.items()} if h is not None else {},
            baseline=self.baseline(), deviations=list(self.deviations),
            risk=self.latest_risk or {},
            observations_used={"first_hour": self._obs[0].hour if self._obs else None,
                               "last_hour": h, "n": self.n_observations})

    def _snapshot(self):
        self._snapshots.append(self.state().to_dict())

    def _snapshot_replace(self):
        if self._snapshots:
            self._snapshots[-1] = self.state().to_dict()

    def state_at(self, hour: int):
        cands = [s for s in self._snapshots if s["hour"] <= hour]
        if not cands:
            raise KeyError(f"No state at or before hour {hour}")
        return cands[-1]
