from datetime import datetime, timedelta

import pytest

from src.services.patient_service import PatientService
from tests.test_dashboard import cfg_in


def register(svc):
    return svc.register_patient({
        "name": "Phase 6 patient", "age": 50, "gender": "Female",
        "conditions": ["Hypertension"],
        "monitoring_start": datetime(2026, 1, 1).isoformat(),
    })


def test_twin_trend_uses_only_sufficient_observed_points_and_marks_nonclinical(tmp_path):
    svc = PatientService(cfg_in(tmp_path), {})
    pid = register(svc)
    start = datetime(2026, 1, 1)
    svc.add_observation(pid, {"HR": 70}, start)
    svc.add_observation(pid, {"HR": 72}, start + timedelta(hours=1))
    sn = svc.snapshot(pid)
    assert sn["health_trajectory"]["available"] is False
    assert sn["health_trajectory"]["clinical_forecast"] is False
    svc.add_observation(pid, {"HR": 74}, start + timedelta(hours=2))
    trend = svc.snapshot(pid)["health_trajectory"]
    assert trend["available"] is True
    assert trend["clinical_forecast"] is False
    assert trend["variables"]["HR"]["direction"] == "increasing"
    assert trend["variables"]["HR"]["slope_per_hour"] == pytest.approx(2.0)
    assert trend["variables"]["HR"]["n_points"] == 3


def test_twin_trend_ignores_missing_values_and_baseline_quality_is_patient_specific(tmp_path):
    svc = PatientService(cfg_in(tmp_path), {})
    pid = register(svc)
    start = datetime(2026, 1, 1)
    for i, hr in enumerate([80, None, 84, 86]):
        values = {"O2Sat": 98}
        if hr is not None:
            values["HR"] = hr
        svc.add_observation(pid, values, start + timedelta(hours=i))
    sn = svc.snapshot(pid)
    assert sn["health_trajectory"]["variables"]["HR"]["n_points"] == 3
    assert sn["health_trajectory"]["variables"]["HR"]["direction"] == "increasing"
    assert sn["baseline_quality"]["status"] == "established"
    assert "this patient's" in sn["baseline_quality"]["note"]


def test_twin_trend_survives_database_restore(tmp_path):
    svc = PatientService(cfg_in(tmp_path), {})
    pid = register(svc)
    start = datetime(2026, 1, 1)
    for i, hr in enumerate([90, 88, 86]):
        svc.add_observation(pid, {"HR": hr}, start + timedelta(hours=i))
    before = svc.snapshot(pid)["health_trajectory"]
    svc2 = PatientService(cfg_in(tmp_path), {})
    after = svc2.snapshot(pid)["health_trajectory"]
    assert after == before
    assert after["variables"]["HR"]["direction"] == "decreasing"
