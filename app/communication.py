import sqlite3
import hashlib
from datetime import datetime

class CommunicationModule:
    def __init__(self, db_path="healthtwin_alerts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, sender TEXT, message TEXT, timestamp TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS call_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, room_url TEXT, created_at TEXT)""")
        conn.commit()
        conn.close()

    def send_message(self, patient_id, sender, message):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO messages (patient_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
                      (patient_id, sender, message, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_messages(self, patient_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT sender, message, timestamp FROM messages WHERE patient_id=? ORDER BY timestamp", (patient_id,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def create_video_call(self, patient_id):
        seed = f"{patient_id}-{datetime.now().isoformat()}"
        room_id = hashlib.sha256(seed.encode()).hexdigest()[:12]
        room_url = f"https://meet.jit.si/healthtwin-{room_id}"
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO call_sessions (patient_id, room_url, created_at) VALUES (?, ?, ?)",
                      (patient_id, room_url, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return room_url

    def trigger_communication(self, patient_id, alert_level):
        if alert_level.value == "urgent_alert":
            link = self.create_video_call(patient_id)
            self.send_message(patient_id, "system", f"Urgent alert triggered. Video consultation ready: {link}")
            return link
        elif alert_level.value == "soft_notify":
            self.send_message(patient_id, "system", "Vitals trending upward — clinician has been notified for monitoring.")
        return None
