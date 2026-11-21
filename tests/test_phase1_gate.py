"""Phase 1: provenance + model availability gate, dependency list, UTF-8 text I/O. SYNTHETIC data only - nothing here is a real-data result."""
import json, re
from pathlib import Path
import pytest
from src.utils.config import load_config
from src.utils.provenance import check_gate, read_manifest, write_manifest, REAL_ORIGINS
from src.simulation.session import load_assets
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from src.models.train_sepsis import run_training
from tests.test_sepsis_phase3 import synth
from tests.test_dashboard import cfg_in

ROOT = Path(__file__).resolve().parents[1]


def train(tmp_path, provenance):
    df = synth(); sp = patient_wise_split(df, seed=1)
    art, rep = tmp_path / "artifacts_dir" / "sepsis", tmp_path / "reports_dir"
    run_training(df, sp, art, rep, 1, provenance=provenance)
    return art, rep / "sepsis_metrics.json"


def test_no_artifact_is_unavailable(tmp_path):
    g = check_gate(tmp_path / "none", tmp_path / "none.json", allow_test=False)
    assert g["state"] == "unavailable" and not g["passed"] and "No trained artifact" in g["reason"]


def test_default_training_provenance_is_unverified_and_not_available(tmp_path):
    art, rep = train(tmp_path, None)
    meta = json.loads((art / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["provenance"]["data_origin"] == "unverified" and meta["trained_at"] and meta["pipeline_version"]
    g = check_gate(art, rep, allow_test=False)
    assert g["state"] == "pipeline_test" and not g["passed"] and "UNVERIFIED" in g["reason"]


def test_synthetic_origin_is_never_available(tmp_path):
    art, rep = train(tmp_path, {"data_origin": "synthetic_test", "source": "unit test"})
    assert not check_gate(art, rep, allow_test=False)["passed"]


def test_real_origin_with_matching_report_is_validated(tmp_path):
    # the origin string is only a label supplied by the caller; this test exercises the gate logic, it does not claim the toy data are real
    art, rep = train(tmp_path, {"data_origin": "real_public", "source": "gate logic test"})
    g = check_gate(art, rep, allow_test=False)
    assert g["passed"] and g["state"] == "validated" and g["origin"] in REAL_ORIGINS


def test_missing_or_mismatched_report_blocks_availability(tmp_path):
    art, rep = train(tmp_path, {"data_origin": "real_public", "source": "gate logic test"})
    data = json.loads(rep.read_text(encoding="utf-8"))
    data["meta"]["trained_at"] = "1999-01-01T00:00:00+00:00"           # report from a different training run
    rep.write_text(json.dumps(data), encoding="utf-8")
    assert not check_gate(art, rep, allow_test=False)["passed"]
    rep.unlink()
    g = check_gate(art, rep, allow_test=False)
    assert not g["passed"] and "report missing" in g["reason"]


def test_pre_phase1_artifact_without_provenance_is_not_available(tmp_path):
    art, rep = train(tmp_path, {"data_origin": "real_public", "source": "x"})
    meta = json.loads((art / "model_meta.json").read_text(encoding="utf-8"))
    for k in ("provenance", "trained_at"): meta.pop(k)
    (art / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert not check_gate(art, rep, allow_test=False)["passed"]


def test_load_assets_refuses_unvalidated_model_unless_test_mode(tmp_path, monkeypatch):
    cfg = cfg_in(tmp_path); train(tmp_path, None)
    monkeypatch.delenv("HEALTHTWIN_ALLOW_TEST_MODELS", raising=False)
    a = load_assets(cfg)
    assert a["model"] is None and a["model_error"] and a["gates"]["sepsis"]["state"] == "pipeline_test"
    monkeypatch.setenv("HEALTHTWIN_ALLOW_TEST_MODELS", "1")
    b = load_assets(cfg)
    assert b["model"] is not None and not b["gates"]["sepsis"]["passed"] and b["gates"]["sepsis"]["reason"].startswith("TEST MODE")


def test_load_assets_loads_validated_model(tmp_path, monkeypatch):
    cfg = cfg_in(tmp_path); train(tmp_path, {"data_origin": "real_public", "source": "gate logic test"})
    monkeypatch.delenv("HEALTHTWIN_ALLOW_TEST_MODELS", raising=False)
    a = load_assets(cfg)
    assert a["model"] is not None and a["gates"]["sepsis"]["passed"]


def test_manifest_roundtrip_and_default(tmp_path):
    assert read_manifest(tmp_path)["data_origin"] == "unverified"
    write_manifest(tmp_path, "user_confirmed_real", "test", n=3)
    m = read_manifest(tmp_path)
    assert m["data_origin"] == "user_confirmed_real" and m["n"] == 3


def test_requirements_cover_parquet_and_plots():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("pyarrow", "matplotlib", "scikit-learn", "streamlit", "shap"):
        assert pkg in req


def test_text_io_specifies_utf8():
    """Windows default text encoding is cp1252: every read_text/write_text/open in text mode must say encoding='utf-8'."""
    bad = []
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + [ROOT / "app.py"]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.(read_text|write_text)\(", line) and "encoding" not in line:
                nxt = f.read_text(encoding="utf-8").splitlines()[i] if i < len(f.read_text(encoding="utf-8").splitlines()) else ""
                if "encoding" not in nxt:
                    bad.append(f"{f.relative_to(ROOT)}:{i}")
    assert bad == [], bad
