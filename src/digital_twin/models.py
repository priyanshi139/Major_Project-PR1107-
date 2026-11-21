from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Observation:
    """One timestamped reading for one subject. `hour` = simulated time index (replay), not wall-clock."""
    subject_id: str
    module: str
    hour: int
    values: dict
    timestamp: Optional[str] = None      # ISO data time (None = relative hour only, e.g. public dataset records)
    source: str = "unspecified"           # historical_dataset | manual_entry | simulated_live | synthetic_demo | uploaded

    def __post_init__(self):
        if ":" not in self.subject_id or self.subject_id.split(":")[0] != self.module:
            raise ValueError(f"subject_id '{self.subject_id}' must be namespaced as '{self.module}:<id>'")
        if not isinstance(self.hour, int) or self.hour < 0:
            raise ValueError("hour must be a non-negative int")


@dataclass
class PatientState:
    subject_id: str
    module: str
    hour: int
    n_observations: int
    last_update_wall_time: float
    status: str
    data_quality: str
    latest_values: dict
    hours_since_measured: dict
    baseline: dict
    deviations: list
    risk: dict
    observations_used: dict
    is_simulated_replay: bool = True

    def to_dict(self):
        return asdict(self)
