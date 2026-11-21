import numpy as np
import pandas as pd
import torch
from src.models.causal_transformer import CausalTransformer, make_patient_windows


def test_transformer_cannot_use_future_tokens():
    torch.manual_seed(7)
    model = CausalTransformer(n_features=3, d_model=16, n_heads=4, n_layers=1,
                              dropout=0.0, max_len=8).eval()
    x = torch.randn(1, 8, 3)
    changed = x.clone(); changed[:, 5:, :] += 1000.0
    with torch.no_grad():
        a = model.forward_all(x)
        b = model.forward_all(changed)
    # Outputs through hour 4 are invariant to changes in hours 5 onward.
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)


def test_windows_are_patient_local_and_chronological():
    frame = pd.DataFrame({
        "subject_id": ["p1"] * 4 + ["p2"] * 3,
        "hour": [0, 1, 2, 3, 0, 1, 2],
        "f": [1, 2, 3, 4, 10, 11, 12],
        "SepsisLabel": [0, 0, 1, 1, 0, 0, 1],
    })
    x, y, ids, hours = make_patient_windows(frame, ["f"], seq_len=3, stride=1)
    assert set(ids) == {"p1", "p2"}
    assert len(x) == len(y) == len(ids) == len(hours)
    # A p1 window cannot contain p2 values (or vice versa).
    for row, pid in zip(x, ids):
        nonzero = row[row[:, 0] != 0, 0]
        assert (nonzero < 10).all() if pid == "p1" else (nonzero >= 10).all()


def test_window_builder_rejects_empty_input():
    frame = pd.DataFrame(columns=["subject_id", "hour", "f", "SepsisLabel"])
    try:
        make_patient_windows(frame, ["f"])
    except ValueError as exc:
        assert "No sequence windows" in str(exc)
    else:
        raise AssertionError("Expected empty input to fail")
