"""SYNTHETIC data only. Tests code paths, not real performance."""
import numpy as np, pandas as pd, pytest, wfdb
from src.forecasting.baselines import (feature_frame, make_supervised, train_forecasters, save_forecasters,
                                       ForecastService)
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from src.ecg.beats import AAMI_MAP, extract_beats, feature_matrix, CLASSES
from src.ecg.train import split_records, run_training_from_beats, DS1, DS2, multiclass_metrics
from src.ecg.loader import list_records, load_record
from src.ecg.inference import EcgBeatModel


def ar_df(n_pat=60, n_h=40, phi=0.5, seed=0, random_walk=False):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n_pat):
        x = [80.0]
        for _ in range(n_h - 1):
            x.append(x[-1] + rng.normal(0, 2) if random_walk else 80 + phi * (x[-1] - 80) + rng.normal(0, 4))
        d = pd.DataFrame({"subject_id": f"sepsis:p{i}", "hour": range(n_h), "HR": x, "ICULOS": range(1, n_h + 1),
                          "SepsisLabel": int(i % 5 == 0)})
        d.loc[d.index[::7], "HR"] = np.nan
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_features_are_causal():
    df = ar_df(3, 20); X1 = feature_frame(df, "HR")
    df2 = df.copy(); df2.loc[df2.hour > 10, "HR"] = 999.0
    X2 = feature_frame(df2, "HR")
    pd.testing.assert_frame_equal(X1[df.hour <= 10].dropna(), X2[df.hour <= 10].dropna())
    X, y = make_supervised(df, "HR", 3)
    assert y.notna().all() and X.notna().all().all()


def test_forecaster_enabled_only_when_beating_persistence(tmp_path):
    for rw, expect in [(False, True), (True, False)]:
        df = ar_df(random_walk=rw, seed=1); sp = patient_wise_split(df, seed=1)
        models, met = train_forecasters(df, sp, ["HR"], [1])
        assert met["HR|1"]["enabled"] is expect, (rw, met["HR|1"])
    df = ar_df(seed=1); sp = patient_wise_split(df, seed=1)
    models, met = train_forecasters(df, sp, ["HR"], [1]); save_forecasters(models, met, tmp_path / "a", tmp_path / "r")
    fs = ForecastService(tmp_path / "a")
    r = fs.predict(df[df.subject_id == "sepsis:p3"].iloc[:15], "HR", 1)
    assert r["available"] and r["lo"] < r["value"] < r["hi"] and r["target_hour"] == 15
    assert not fs.predict(df.iloc[:2], "HR", 1)["available"]          # too little history
    assert "reason" in fs.predict(df.iloc[:15], "Resp", 1)
    with pytest.raises(FileNotFoundError): ForecastService(tmp_path / "none")


def test_forecast_too_little_data_disabled():
    df = ar_df(5, 10); sp = patient_wise_split(df, seed=1)
    _, met = train_forecasters(df, sp, ["HR"], [1])
    assert met["HR|1"]["enabled"] is False


# ---------------- ECG ----------------
def synth_record(n_beats=60, fs=360, seed=0, classes=("N", "N", "N", "V", "S", "F")):
    """SYNTHETIC ECG-like record: V = tall beat, F = medium beat, S = premature beat (short pre-RR, long pause after)."""
    rng = np.random.default_rng(seed); amp = {"N": 1.0, "S": 1.0, "V": 2.0, "F": 1.5}
    pos, samples, symbols = 200, [], []
    for k in range(n_beats):
        sym = classes[k % len(classes)]
        pos += int((0.45 if sym == "S" else (1.15 if classes[(k - 1) % len(classes)] == "S" else 0.8)) * fs)
        samples.append(pos); symbols.append(sym)
    sig = rng.normal(0, 0.02, samples[-1] + 400)
    for s_, sym in zip(samples, symbols):
        sig[s_ - 3:s_ + 3] += amp[sym]
    return sig, np.array(samples), symbols


def test_extract_beats_and_mapping():
    sig, smp, sym = synth_record(); sym2 = sym[:]; sym2[1] = "/"       # paced beat must be excluded
    b = extract_beats(sig, smp, sym2)
    assert b["waves"].shape[1] == 200 and b["rr"].shape == (len(b["labels"]), 4)
    assert len(b["labels"]) == len(smp) - 1 and set(b["labels"]) <= set(CLASSES)
    assert AAMI_MAP["L"] == "N" and "/" not in AAMI_MAP
    assert feature_matrix(b["waves"], b["rr"]).shape[0] == len(b["labels"])
    e = extract_beats(sig, np.array([5, 100000]), ["N", "N"]); assert len(e["labels"]) == 0   # out of bounds skipped


def test_record_split_is_disjoint_and_record_wise():
    sp = split_records(DS1 + DS2)
    allr = sp["train"] + sp["val"] + sp["test"]
    assert len(allr) == len(set(allr)) and not set(DS1) & set(DS2)
    assert set(sp["test"]) == set(DS2) and not set(sp["test"]) & set(sp["train"])


def test_ecg_training_pipeline_synthetic(tmp_path):
    beats = {r: extract_beats(*synth_record(seed=i)) for i, r in enumerate(DS1 + DS2)}
    rep = run_training_from_beats(beats, tmp_path / "a", tmp_path / "r", seed=1)
    assert (tmp_path / "a" / "model.joblib").exists() and (tmp_path / "r" / "ecg_metrics.json").exists()
    assert rep["results"]["baseline_always_N"]["test"]["balanced_accuracy"] < 0.5
    assert rep["results"][rep["meta"]["selected_model"]]["test"]["macro_f1_present_classes"] > 0.8
    m = EcgBeatModel(tmp_path / "a")
    sig, smp, sym = synth_record(seed=99)
    out = m.classify_record({"signal": sig, "samples": smp, "symbols": sym, "fs": 360})
    assert len(out["pred"]) == len(out["true"]) > 0
    with pytest.raises(ValueError): run_training_from_beats({"101": beats["101"]}, tmp_path / "x", tmp_path / "y")
    with pytest.raises(FileNotFoundError): EcgBeatModel(tmp_path / "missing")


def test_multiclass_metrics_handles_missing_class():
    m = multiclass_metrics(np.array(["N", "N", "V"]), np.array(["N", "V", "V"]))
    assert m["per_class"]["F"]["support"] == 0 and 0 <= m["balanced_accuracy"] <= 1


def test_wfdb_loader_roundtrip(tmp_path):
    sig, smp, sym = synth_record(20)
    wfdb.wrsamp("101", fs=360, units=["mV"], sig_name=["MLII"], p_signal=sig.reshape(-1, 1), fmt=["16"], write_dir=str(tmp_path))
    wfdb.wrann("101", "atr", smp, sym, write_dir=str(tmp_path))
    assert list_records(tmp_path) == ["101"]
    r = load_record(tmp_path, "101")
    assert r["fs"] == 360 and len(r["samples"]) == 20 and r["symbols"][0] == "N" and r["lead"] == "MLII"
    with pytest.raises(FileNotFoundError): load_record(tmp_path, "999")
    assert list_records(tmp_path / "nope") == []
