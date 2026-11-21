import math
import numpy as np


class RollingBaselineDetector:
    """Flags a reading that is unusual relative to THIS patient's own recent past (past values only).
    An unusual reading is not a diagnosis. A patient whose normal is high is not flagged for being high."""

    def __init__(self, z_threshold=3.0, window=24, min_points=6, rel_floor=0.03):
        self.z, self.window, self.min_points, self.rel_floor = z_threshold, window, min_points, rel_floor

    def detect(self, twin, obs) -> list:
        out = []
        for var in twin.vitals:
            x = obs.values.get(var)
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue                                   # missing -> never flagged
            past = twin.past_measured(var, self.window, before_hour=obs.hour)
            if len(past) < self.min_points:
                continue                                   # baseline not established yet
            mean, sd = float(np.mean(past)), float(np.std(past, ddof=1))
            sd_eff = max(sd, self.rel_floor * abs(mean), 1e-6)
            z = (float(x) - mean) / sd_eff
            if abs(z) >= self.z:
                direction = "above" if z > 0 else "below"
                out.append({"variable": var, "value": float(x), "baseline_mean": mean,
                            "baseline_sd": sd_eff, "z": float(z), "direction": direction,
                            "explanation": (f"{var}={float(x):.1f} is {abs(z):.1f} SD {direction} this patient's own "
                                            f"recent baseline (mean {mean:.1f}, last {len(past)} readings). "
                                            f"Unusual reading, not a confirmed medical event.")})
        return sorted(out, key=lambda d: -abs(d["z"]))
