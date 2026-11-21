"""Data provenance and the model availability gate.

A model is shown as AVAILABLE only when ALL of these hold:
  1. its trained artifact exists and loads,
  2. its model_meta.json records the data provenance and a training timestamp,
  3. the provenance says the data are real public data (see REAL_ORIGINS),
  4. an evaluation report exists and belongs to THIS artifact (same trained_at) and contains test results.
Anything else is 'unavailable', or 'pipeline_test' (artifact loads but is not validated, e.g. trained on synthetic or unverified files).
Pipeline-test models are never loaded by the app unless HEALTHTWIN_ALLOW_TEST_MODELS=1 is set (used by developers)."""
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_VERSION = "v4.1-phase1"
MANIFEST = "PROVENANCE.json"
# real_public          : downloaded by this project's download script from the public source
# user_confirmed_real  : user supplied/confirmed the unmodified public files (--zip or --confirm-physionet-origin)
REAL_ORIGINS = {"real_public", "user_confirmed_real"}
ORIGIN_LABELS = {"real_public": "real public data (downloaded by the project script)",
                 "user_confirmed_real": "real public data (confirmed by the user)",
                 "unverified": "UNVERIFIED origin", "synthetic_test": "SYNTHETIC test data"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(folder, origin: str, source: str, **extra) -> dict:
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    m = {"data_origin": origin, "source": source, "recorded_at": now_utc(), **extra}
    (folder / MANIFEST).write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def read_manifest(folder) -> dict:
    try:
        return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    except Exception:
        return {"data_origin": "unverified", "source": "no PROVENANCE.json next to the data"}


def allow_test_models() -> bool:
    return os.environ.get("HEALTHTWIN_ALLOW_TEST_MODELS") == "1"


def check_gate(artifacts_dir, report_path, allow_test=None) -> dict:
    """-> {state: validated|pipeline_test|unavailable, passed, reason, origin, trained_at}. Never raises."""
    allow_test = allow_test_models() if allow_test is None else allow_test
    out = {"state": "unavailable", "passed": False, "reason": "", "origin": None, "trained_at": None}
    meta_p = Path(artifacts_dir) / "model_meta.json"
    if not meta_p.exists():
        out["reason"] = "No trained artifact found (model_meta.json missing)."
        return out
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as e:
        out["reason"] = f"model_meta.json unreadable: {e}"
        return out
    prov = meta.get("provenance") or {}
    out["origin"], out["trained_at"] = prov.get("data_origin"), meta.get("trained_at")
    if not prov or not meta.get("trained_at"):
        out.update(state="pipeline_test", reason="Artifact has no recorded data provenance / training time (trained before Phase 1 or by an old script). "
                                                  "Re-run the training script to make it verifiable.")
        return _maybe_test(out, allow_test)
    rp = Path(report_path)
    if not rp.exists():
        out.update(state="pipeline_test", reason=f"Evaluation report missing ({rp.name}); a model without an evaluation is not shown as available.")
        return _maybe_test(out, allow_test)
    try:
        rep = json.loads(rp.read_text(encoding="utf-8"))
    except Exception as e:
        out.update(state="pipeline_test", reason=f"Evaluation report unreadable: {e}")
        return _maybe_test(out, allow_test)
    rep_trained = (rep.get("meta") or {}).get("trained_at") or rep.get("trained_at")
    if rep_trained != meta.get("trained_at") or not rep.get("results"):
        out.update(state="pipeline_test", reason="Evaluation report does not belong to this artifact (training time differs) or has no results.")
        return _maybe_test(out, allow_test)
    if prov.get("data_origin") not in REAL_ORIGINS:
        out.update(state="pipeline_test", reason=f"Trained on {ORIGIN_LABELS.get(prov.get('data_origin'), 'data of unknown origin')}; "
                                                  "performance is not real-data performance and the model is not shown as available.")
        return _maybe_test(out, allow_test)
    out.update(state="validated", passed=True, reason="")
    return out


def _maybe_test(out, allow_test):
    out["test_mode_loadable"] = bool(allow_test)
    return out
