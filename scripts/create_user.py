"""Create a doctor/admin/patient account in the app database (clinician accounts are not self-service).
  python scripts/create_user.py --username drmehta --role doctor --password "a-long-password"
Patient accounts can also be linked to an existing patient: --patient-id HT-0001"""
import argparse, getpass, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path
from src.database.patient_store import PatientStore
from src.api.security import hash_password


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True); ap.add_argument("--role", choices=["doctor", "admin", "patient"], required=True)
    ap.add_argument("--password"); ap.add_argument("--full-name", default=""); ap.add_argument("--patient-id")
    a = ap.parse_args()
    pw = a.password or getpass.getpass("Password (min 8 chars): ")
    if len(pw) < 8: sys.exit("Password must be at least 8 characters")
    store = PatientStore(resolve_path(load_config(), "artifacts_dir") / "patients.db")
    if store.get_user_by_name(a.username): sys.exit("Username already exists")
    if a.patient_id and store.get_patient(a.patient_id) is None: sys.exit(f"Unknown patient {a.patient_id}")
    uid = store.create_user(a.username, hash_password(pw), a.role, a.full_name, a.patient_id)
    store.audit("create_user.py", None, "user_created", a.patient_id, {"username": a.username, "role": a.role})
    print(f"Created {a.role} '{a.username}' (user id {uid}).")


if __name__ == "__main__":
    main()
