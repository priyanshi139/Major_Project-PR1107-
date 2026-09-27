import sqlite3
from datetime import datetime
from enum import Enum

class AlertLevel(Enum):
    NONE = "none"
    SOFT = "soft_notify"
    URGENT = "urgent_alert"

class HealthTwinAgent:
    def __init__(self, urgent_threshold=0.7, soft_threshold=0.5, db_path="healthtwin_alerts.db"):
        self.urgent_threshold = urgent_threshold
        self.soft_threshold = soft_threshold
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, timestamp TEXT, risk_score REAL,
            alert_level TEXT, message TEXT)""")
        conn.commit()
        conn.close()

    def classify(self, risk_score):
        if risk_score >= self.urgent_threshold:
            return AlertLevel.URGENT
        elif risk_score >= self.soft_threshold:
            return AlertLevel.SOFT
        return AlertLevel.NONE

    def _generate_message(self, risk_score, level):
        if level == AlertLevel.URGENT:
            return f"Risk score {risk_score:.2f} — deterioration likely within hours. Immediate review recommended."
        elif level == AlertLevel.SOFT:
            return f"Risk score {risk_score:.2f} — trending upward. Keep monitoring."
        return "No action needed."

    def log_alert(self, patient_id, risk_score, level):
        message = self._generate_message(risk_score, level)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO alerts (patient_id, timestamp, risk_score, alert_level, message) VALUES (?, ?, ?, ?, ?)",
            (patient_id, datetime.now().isoformat(), risk_score, level.value, message)
        )
        conn.commit()
        conn.close()
        return message

    def get_alert_history(self, patient_id=None, limit=50):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if patient_id:
            cur.execute("SELECT timestamp, risk_score, alert_level, message FROM alerts WHERE patient_id=? ORDER BY timestamp DESC LIMIT ?", (patient_id, limit))
        else:
            cur.execute("SELECT timestamp, risk_score, alert_level, message FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
