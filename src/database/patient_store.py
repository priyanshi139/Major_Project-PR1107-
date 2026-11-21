import json, time
from src.database.db import Database, _clean


class PatientStore(Database):
    """Database + patient profiles, observation metadata (timestamp/source), timeline events and what-if runs.
    Implements the same save_* interface used by TwinEngine/AlertManager, so they work unchanged."""

    def __init__(self, path=":memory:"):
        super().__init__(path)
        with self._lock:
            self._c.executescript("""
            CREATE TABLE IF NOT EXISTS patients(patient_id TEXT PRIMARY KEY, payload TEXT, created_wall REAL, updated_wall REAL);
            CREATE TABLE IF NOT EXISTS obs_meta(subject_id TEXT, hour INTEGER, timestamp TEXT, source TEXT, PRIMARY KEY(subject_id, hour));
            CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, hour INTEGER, timestamp TEXT,
                wall_time REAL, event_type TEXT, title TEXT, detail TEXT, severity TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS whatif_runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, wall_time REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS disease_assessments(assessment_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                disease TEXT, hour INTEGER, wall_time REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient','doctor','admin')), full_name TEXT, patient_id TEXT, settings TEXT DEFAULT '{}',
                created_wall REAL, last_login_wall REAL);
            CREATE TABLE IF NOT EXISTS proposals(proposal_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, alert_id TEXT, disease TEXT,
                title TEXT, rationale TEXT, evidence TEXT, status TEXT NOT NULL DEFAULT 'proposed', created_wall REAL, decided_by TEXT, decided_wall REAL, decision_note TEXT);
            CREATE TABLE IF NOT EXISTS audit_log(audit_id INTEGER PRIMARY KEY AUTOINCREMENT, wall_time REAL, username TEXT, role TEXT, action TEXT, patient_id TEXT, detail TEXT);
            """)
            self._c.commit()

    # ---- patients ----
    def next_patient_id(self) -> str:
        with self._lock:
            ids = [r[0] for r in self._c.execute("SELECT patient_id FROM patients").fetchall()]
        nums = [int(i.split("-")[1]) for i in ids if i.startswith("HT-") and i.split("-")[1].isdigit()]
        return f"HT-{max(nums, default=0) + 1:04d}"

    def create_patient(self, profile: dict) -> str:
        pid = self.next_patient_id(); now = time.time()
        with self._lock:
            self._c.execute("INSERT INTO patients VALUES (?,?,?,?)", (pid, json.dumps(_clean(profile), default=str), now, now))
            self._c.commit()
        return pid

    def update_patient(self, pid: str, profile: dict):
        with self._lock:
            cur = self._c.execute("UPDATE patients SET payload=?, updated_wall=? WHERE patient_id=?",
                                  (json.dumps(_clean(profile), default=str), time.time(), pid))
            self._c.commit()
        if cur.rowcount == 0:
            raise KeyError(f"Unknown patient {pid}")

    def get_patient(self, pid: str):
        with self._lock:
            r = self._c.execute("SELECT payload, created_wall, updated_wall FROM patients WHERE patient_id=?", (pid,)).fetchone()
        if r is None:
            return None
        return {**json.loads(r[0]), "patient_id": pid, "created_wall": r[1], "updated_wall": r[2]}

    def list_patients(self) -> list:
        with self._lock:
            ids = [r[0] for r in self._c.execute("SELECT patient_id FROM patients ORDER BY patient_id").fetchall()]
        return [self.get_patient(i) for i in ids]

    def delete_patient(self, pid: str):
        self.clear_subject(f"sepsis:{pid}")
        with self._lock:
            for t in ("events", "whatif_runs", "disease_assessments", "proposals"):
                self._c.execute(f"DELETE FROM {t} WHERE patient_id=?", (pid,))
            self._c.execute("UPDATE users SET patient_id=NULL WHERE patient_id=?", (pid,))
            self._c.execute("DELETE FROM patients WHERE patient_id=?", (pid,))
            self._c.commit()

    # ---- observation metadata ----
    def save_observation(self, obs):
        super().save_observation(obs)
        with self._lock:
            self._c.execute("INSERT OR REPLACE INTO obs_meta VALUES (?,?,?,?)", (obs.subject_id, int(obs.hour), obs.timestamp, obs.source))
            self._c.commit()

    def get_obs_meta(self, subject_id: str) -> dict:
        with self._lock:
            rows = self._c.execute("SELECT hour, timestamp, source FROM obs_meta WHERE subject_id=?", (subject_id,)).fetchall()
        return {h: {"timestamp": t, "source": s} for h, t, s in rows}

    def clear_subject(self, subject_id):
        super().clear_subject(subject_id)
        with self._lock:
            self._c.execute("DELETE FROM obs_meta WHERE subject_id=?", (subject_id,)); self._c.commit()

    # ---- timeline ----
    def add_event(self, patient_id, event_type, title, detail=None, hour=None, timestamp=None, severity=None, source=None) -> int:
        with self._lock:
            cur = self._c.execute("INSERT INTO events(patient_id,hour,timestamp,wall_time,event_type,title,detail,severity,source) VALUES (?,?,?,?,?,?,?,?,?)",
                                  (patient_id, hour, timestamp, time.time(), event_type, title,
                                   json.dumps(_clean(detail or {}), default=str), severity, source))
            self._c.commit()
            return cur.lastrowid

    def get_events(self, patient_id=None, types=None, limit=None) -> list:
        q, a = "SELECT event_id,patient_id,hour,timestamp,wall_time,event_type,title,detail,severity,source FROM events WHERE 1=1", []
        if patient_id: q += " AND patient_id=?"; a.append(patient_id)
        if types: q += f" AND event_type IN ({','.join('?' * len(types))})"; a += list(types)
        q += " ORDER BY event_id" + (" DESC LIMIT ?" if limit else "")
        if limit: a.append(limit)
        with self._lock:
            rows = self._c.execute(q, a).fetchall()
        out = [{"event_id": r[0], "patient_id": r[1], "hour": r[2], "timestamp": r[3], "wall_time": r[4], "event_type": r[5], "title": r[6],
                "detail": json.loads(r[7]), "severity": r[8], "source": r[9]} for r in rows]
        return out[::-1] if limit else out

    # ---- what-if runs (kept apart from observations) ----
    def save_whatif(self, patient_id, payload: dict) -> int:
        with self._lock:
            cur = self._c.execute("INSERT INTO whatif_runs(patient_id,wall_time,payload) VALUES (?,?,?)",
                                  (patient_id, time.time(), json.dumps(_clean(payload), default=str)))
            self._c.commit(); return cur.lastrowid

    def get_whatifs(self, patient_id) -> list:
        with self._lock:
            rows = self._c.execute("SELECT run_id, wall_time, payload FROM whatif_runs WHERE patient_id=? ORDER BY run_id", (patient_id,)).fetchall()
        return [{"run_id": r, "wall_time": w, **json.loads(p)} for r, w, p in rows]


    # ---- patient-linked non-temporal disease assessments ----
    def save_disease_assessment(self, patient_id, disease, hour, payload):
        with self._lock:
            cur = self._c.execute(
                "INSERT INTO disease_assessments(patient_id,disease,hour,wall_time,payload) VALUES (?,?,?,?,?)",
                (patient_id, disease, int(hour), time.time(), json.dumps(_clean(payload), default=str)))
            self._c.commit()
            return cur.lastrowid

    def get_disease_assessments(self, patient_id, disease=None):
        q, args = "SELECT assessment_id,disease,hour,wall_time,payload FROM disease_assessments WHERE patient_id=?", [patient_id]
        if disease:
            q += " AND disease=?"; args.append(disease)
        q += " ORDER BY assessment_id"
        with self._lock:
            rows = self._c.execute(q, args).fetchall()
        return [{"assessment_id": r[0], "disease": r[1], "hour": r[2], "wall_time": r[3], **json.loads(r[4])} for r in rows]

    # ---------------- users ----------------
    def create_user(self, username, password_hash, role, full_name="", patient_id=None) -> int:
        with self._lock:
            cur = self._c.execute("INSERT INTO users(username,password_hash,role,full_name,patient_id,created_wall) VALUES (?,?,?,?,?,?)",
                                  (username, password_hash, role, full_name, patient_id, time.time()))
            self._c.commit(); return cur.lastrowid

    def _user_row(self, r):
        if r is None:
            return None
        return {"user_id": r[0], "username": r[1], "password_hash": r[2], "role": r[3], "full_name": r[4], "patient_id": r[5],
                "settings": json.loads(r[6] or "{}"), "created_wall": r[7], "last_login_wall": r[8]}

    def get_user_by_name(self, username):
        with self._lock:
            r = self._c.execute("SELECT user_id,username,password_hash,role,full_name,patient_id,settings,created_wall,last_login_wall FROM users WHERE username=?", (username,)).fetchone()
        return self._user_row(r)

    def get_user(self, user_id):
        with self._lock:
            r = self._c.execute("SELECT user_id,username,password_hash,role,full_name,patient_id,settings,created_wall,last_login_wall FROM users WHERE user_id=?", (user_id,)).fetchone()
        return self._user_row(r)

    def list_users(self):
        with self._lock:
            rows = self._c.execute("SELECT user_id,username,password_hash,role,full_name,patient_id,settings,created_wall,last_login_wall FROM users ORDER BY user_id").fetchall()
        return [self._user_row(r) for r in rows]

    def touch_login(self, user_id):
        with self._lock:
            self._c.execute("UPDATE users SET last_login_wall=? WHERE user_id=?", (time.time(), user_id)); self._c.commit()

    def set_user_settings(self, user_id, settings: dict):
        with self._lock:
            self._c.execute("UPDATE users SET settings=? WHERE user_id=?", (json.dumps(settings), user_id)); self._c.commit()

    def link_user_patient(self, user_id, patient_id):
        with self._lock:
            self._c.execute("UPDATE users SET patient_id=? WHERE user_id=?", (patient_id, user_id)); self._c.commit()

    def delete_user(self, user_id):
        with self._lock:
            self._c.execute("DELETE FROM users WHERE user_id=?", (user_id,)); self._c.commit()

    # ---------------- agent proposals (human approval required) ----------------
    def add_proposal(self, patient_id, alert_id, disease, title, rationale, evidence) -> int:
        with self._lock:
            cur = self._c.execute("INSERT INTO proposals(patient_id,alert_id,disease,title,rationale,evidence,created_wall) VALUES (?,?,?,?,?,?,?)",
                                  (patient_id, alert_id, disease, title, rationale, json.dumps(_clean(evidence), default=str), time.time()))
            self._c.commit(); return cur.lastrowid

    def _prop(self, r):
        return {"proposal_id": r[0], "patient_id": r[1], "alert_id": r[2], "disease": r[3], "title": r[4], "rationale": r[5],
                "evidence": json.loads(r[6] or "[]"), "status": r[7], "created_wall": r[8], "decided_by": r[9], "decided_wall": r[10], "decision_note": r[11]}

    def get_proposals(self, patient_id=None, status=None) -> list:
        q, a = "SELECT proposal_id,patient_id,alert_id,disease,title,rationale,evidence,status,created_wall,decided_by,decided_wall,decision_note FROM proposals WHERE 1=1", []
        if patient_id: q += " AND patient_id=?"; a.append(patient_id)
        if status: q += " AND status=?"; a.append(status)
        with self._lock:
            rows = self._c.execute(q + " ORDER BY proposal_id DESC", a).fetchall()
        return [self._prop(r) for r in rows]

    def decide_proposal(self, proposal_id, status, by, note=""):
        if status not in ("approved", "rejected"):
            raise ValueError("Decision must be 'approved' or 'rejected'")
        with self._lock:
            cur = self._c.execute("UPDATE proposals SET status=?, decided_by=?, decided_wall=?, decision_note=? WHERE proposal_id=? AND status='proposed'",
                                  (status, by, time.time(), note, proposal_id))
            self._c.commit()
        if cur.rowcount == 0:
            raise ValueError("Proposal not found or already decided")

    # ---------------- audit ----------------
    def audit(self, username, role, action, patient_id=None, detail=None):
        with self._lock:
            self._c.execute("INSERT INTO audit_log(wall_time,username,role,action,patient_id,detail) VALUES (?,?,?,?,?,?)",
                            (time.time(), username, role, action, patient_id, json.dumps(_clean(detail or {}), default=str)))
            self._c.commit()

    def get_audit(self, patient_id=None, limit=200) -> list:
        q, a = "SELECT audit_id,wall_time,username,role,action,patient_id,detail FROM audit_log WHERE 1=1", []
        if patient_id: q += " AND patient_id=?"; a.append(patient_id)
        with self._lock:
            rows = self._c.execute(q + " ORDER BY audit_id DESC LIMIT ?", a + [limit]).fetchall()
        return [{"audit_id": r[0], "wall_time": r[1], "username": r[2], "role": r[3], "action": r[4], "patient_id": r[5], "detail": json.loads(r[6] or "{}")} for r in rows]
