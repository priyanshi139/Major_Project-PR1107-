import json, math, sqlite3, threading
from pathlib import Path


def _clean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class Database:
    def __init__(self, path=":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._c.executescript("""
            CREATE TABLE IF NOT EXISTS observations(subject_id TEXT, module TEXT, hour INTEGER, payload TEXT,
                PRIMARY KEY(subject_id, hour));
            CREATE TABLE IF NOT EXISTS states(subject_id TEXT, module TEXT, hour INTEGER, payload TEXT,
                PRIMARY KEY(subject_id, hour));
            CREATE TABLE IF NOT EXISTS risks(subject_id TEXT, module TEXT, hour INTEGER, payload TEXT,
                PRIMARY KEY(subject_id, hour));
            CREATE TABLE IF NOT EXISTS alerts(alert_id TEXT PRIMARY KEY, subject_id TEXT, module TEXT, hour INTEGER,
                type TEXT, severity TEXT, status TEXT, payload TEXT);
            """)
            self._c.commit()

    def _put(self, table, subject_id, module, hour, payload):
        with self._lock:
            self._c.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?)",
                            (subject_id, module, int(hour), json.dumps(_clean(payload), default=str)))
            self._c.commit()

    def save_observation(self, obs):
        self._put("observations", obs.subject_id, obs.module, obs.hour, obs.values)

    def save_state(self, state: dict):
        self._put("states", state["subject_id"], state["module"], state["hour"], state)

    def save_risk(self, subject_id, module, hour, risk: dict):
        self._put("risks", subject_id, module, hour, risk)

    def save_alert(self, alert: dict):
        with self._lock:
            self._c.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?)",
                            (alert["alert_id"], alert["subject_id"], alert["module"], alert["hour"],
                             alert["type"], alert["severity"], alert["status"],
                             json.dumps(_clean(alert), default=str)))
            self._c.commit()

    def _get(self, table, subject_id):
        with self._lock:
            rows = self._c.execute(f"SELECT hour, payload FROM {table} WHERE subject_id=? ORDER BY hour",
                                   (subject_id,)).fetchall()
        return [{"hour": h, **({"values": json.loads(p)} if table == "observations" else json.loads(p))}
                for h, p in rows]

    def get_observations(self, subject_id): return self._get("observations", subject_id)
    def get_states(self, subject_id): return self._get("states", subject_id)
    def get_risks(self, subject_id): return self._get("risks", subject_id)

    def get_alerts(self, subject_id=None, status=None):
        q, a = "SELECT payload FROM alerts WHERE 1=1", []
        if subject_id: q += " AND subject_id=?"; a.append(subject_id)
        if status: q += " AND status=?"; a.append(status)
        with self._lock:
            rows = self._c.execute(q + " ORDER BY hour, alert_id", a).fetchall()
        return [json.loads(r[0]) for r in rows]

    def clear_subject(self, subject_id):
        with self._lock:
            for t in ("observations", "states", "risks", "alerts"):
                self._c.execute(f"DELETE FROM {t} WHERE subject_id=?", (subject_id,))
            self._c.commit()

    def counts(self):
        with self._lock:
            return {t: self._c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("observations", "states", "risks", "alerts")}
