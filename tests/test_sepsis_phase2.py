"""All data here is SYNTHETIC, generated only to test code paths. Not real patient data."""
import numpy as np
import pandas as pd
import pytest
from src.data_ingestion import sepsis_loader as L
from src.preprocessing import sepsis_preprocessing as P


def make_psv(dirpath, pid, n_hours=10, septic=False, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n_hours, len(L.EXPECTED_COLUMNS))), columns=L.EXPECTED_COLUMNS)
    d["SepsisLabel"] = 0
    if septic:
        d.loc[n_hours - 3:, "SepsisLabel"] = 1
    d.loc[::2, "Lactate"] = np.nan
    d.to_csv(dirpath / f"{pid}.psv", sep="|", index=False)


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "training_setA"
    d.mkdir()
    for i in range(40):
        make_psv(d, f"p{i:06d}", septic=(i % 5 == 0), seed=i)
    return tmp_path


def test_loader_namespacing_and_order(raw_dir):
    df = L.load_all(raw_dir)
    assert df["subject_id"].str.startswith("sepsis:").all()
    assert df["subject_id"].nunique() == 40
    assert (df.groupby("subject_id")["hour"].apply(lambda s: s.is_monotonic_increasing)).all()
    assert (df["source_set"] == "A").all()


def test_loader_missing_dir_and_bad_columns(tmp_path):
    with pytest.raises(FileNotFoundError):
        L.load_all(tmp_path / "nope")
    bad = tmp_path / "bad.psv"
    pd.DataFrame({"HR": [1]}).to_csv(bad, sep="|", index=False)
    with pytest.raises(ValueError):
        L.load_patient_file(bad)


def test_split_no_overlap_and_complete(raw_dir):
    df = L.load_all(raw_dir)
    s = P.patient_wise_split(df, seed=1)
    all_ids = s["train"] + s["val"] + s["test"]
    assert len(all_ids) == len(set(all_ids)) == 40
    assert P.check_split_integrity(s)


def test_split_stratified_and_reproducible(raw_dir):
    df = L.load_all(raw_dir)
    s1, s2 = P.patient_wise_split(df, seed=7), P.patient_wise_split(df, seed=7)
    assert s1 == s2
    out = P.patient_outcomes(df)
    assert out.loc[s1["train"]].sum() > 0 and out.loc[s1["test"]].sum() > 0


def test_split_overlap_detected():
    with pytest.raises(ValueError):
        P.check_split_integrity({"train": ["a", "b"], "val": ["b"], "test": ["c"]})


def test_bad_fractions(raw_dir):
    with pytest.raises(ValueError):
        P.patient_wise_split(L.load_all(raw_dir), 0.5, 0.5, 0.5)


def test_ffill_is_causal_and_per_patient():
    df = pd.DataFrame({"subject_id": ["a", "a", "a", "b", "b"], "hour": [0, 1, 2, 0, 1],
                       **{c: np.nan for c in L.FEATURES}})
    df.loc[0, "HR"] = 80.0   # patient a, hour 0
    df.loc[4, "HR"] = 99.0   # patient b, hour 1 (future for b hour 0)
    out = P.causal_forward_fill(df).set_index(["subject_id", "hour"])
    assert out.loc[("a", 2), "HR"] == 80.0
    assert np.isnan(out.loc[("b", 0), "HR"])      # not filled from the future
    assert out.loc[("b", 1), "HR"] == 99.0        # no leak from patient a


def test_preprocessor_fit_on_train_only(raw_dir):
    df = L.load_all(raw_dir)
    s = P.patient_wise_split(df, seed=3)
    tr = df[df.subject_id.isin(s["train"])]
    te = df[df.subject_id.isin(s["test"])].copy()
    pre = P.SepsisPreprocessor().fit(tr)
    before = pre.means_.copy()
    te["HR"] = te["HR"] + 1000          # extreme test shift must not change fitted stats
    X = pre.transform(te)
    assert pre.means_.equals(before)
    assert X.drop(columns=["subject_id", "SepsisLabel"]).isna().sum().sum() == 0
    assert "Lactate_missing" in X.columns
    assert len(X) == len(te)


def test_transform_before_fit_and_missing_cols(raw_dir):
    df = L.load_all(raw_dir)
    with pytest.raises(RuntimeError):
        P.SepsisPreprocessor().transform(df)
    pre = P.SepsisPreprocessor().fit(df)
    with pytest.raises(ValueError):
        pre.transform(df.drop(columns=["HR"]))
