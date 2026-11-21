from src.digital_twin.twin import DigitalTwin
from src.utils.logging_utils import get_logger

log = get_logger("twin_engine")
_NOT_OBSERVATIONS = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}   # profile/administrative fields


class TwinEngine:
    """historical observation -> Twin update -> model inference -> anomaly detection -> alerts -> database."""

    def __init__(self, module, variables, vitals, model=None, detector=None, alert_manager=None, db=None):
        self.module, self.variables, self.vitals = module, list(variables), list(vitals)
        self.model, self.detector, self.alerts, self.db = model, detector, alert_manager, db
        self.twins = {}

    def get_twin(self, subject_id):
        if subject_id not in self.twins:
            self.twins[subject_id] = DigitalTwin(subject_id, self.module, self.variables, self.vitals)
        return self.twins[subject_id]

    def process(self, obs) -> dict:
        if obs.module != self.module:
            raise ValueError(f"Engine for module '{self.module}' cannot process '{obs.module}' data")
        twin = self.get_twin(obs.subject_id)
        twin.add_observation(obs)                                   # raises on out-of-order
        if self.db:
            self.db.save_observation(obs)
        anomalies = self.detector.detect(twin, obs) if self.detector else []
        twin.record_deviations(anomalies)
        risk, risk_error = None, None
        if self.model is not None:
            try:
                risk = self.model.predict(twin.history_df())        # only observations received so far
                twin.record_risk(risk)
                if self.db:
                    self.db.save_risk(obs.subject_id, obs.module, obs.hour, risk)
            except Exception as e:                                  # never crash the replay on a model error
                risk_error = str(e)
                log.error("Model inference failed at hour %s: %s", obs.hour, e)
        ctx = {"values": {k: v for k, v in obs.values.items() if v is not None and k not in _NOT_OBSERVATIONS},
               "timestamp": obs.timestamp, "source": obs.source}
        rh = [r["risk_score"] for r in twin.risk_history if r.get("risk_score") is not None]
        new_alerts = (self.alerts.evaluate(obs.subject_id, obs.module, obs.hour, risk, anomalies, risk_history=rh, context=ctx)
                      if self.alerts else [])
        state = twin.state().to_dict()
        if self.db:
            self.db.save_state(state)
        return {"state": state, "risk": risk, "risk_error": risk_error, "anomalies": anomalies,
                "new_alerts": new_alerts, "model_available": self.model is not None}

    def reset_subject(self, subject_id):
        if subject_id in self.twins:
            self.twins[subject_id].reset()
        if self.alerts:
            self.alerts.reset_subject(subject_id)
        if self.db:
            self.db.clear_subject(subject_id)
