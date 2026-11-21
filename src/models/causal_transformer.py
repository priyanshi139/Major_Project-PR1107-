"""Causal sequence models for hourly sepsis risk prediction.

Each sample is a fixed-length history ending at the prediction hour. The upper-
triangular attention mask prevents a token from attending to later tokens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CausalTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, max_len: int = 24):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.max_len = max_len

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] > self.max_len:
            raise ValueError(f"Expected [batch, seq<= {self.max_len}, features]")
        return self.forward_all(x)[:, -1]

    def forward_all(self, x):
        if x.ndim != 3 or x.shape[1] > self.max_len:
            raise ValueError(f"Expected [batch, seq<= {self.max_len}, features]")
        z = self.input_projection(x) + self.position[:, :x.shape[1], :]
        # True entries are masked: query position i cannot attend to key j > i.
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device,
                                     dtype=torch.bool), diagonal=1)
        z = self.encoder(z, mask=mask)
        return self.head(self.norm(z)).squeeze(-1)


class SequenceLSTM(nn.Module):
    """Non-attention recurrent comparator with the same input/output contract."""
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def make_patient_windows(frame, feature_cols, label_col="SepsisLabel", seq_len=24,
                         stride=3, max_windows_per_patient=None):
    """Create chronological, within-patient windows; never crosses patient boundaries."""
    xs, ys, ids, hours = [], [], [], []
    for pid, group in frame.sort_values(["subject_id", "hour"]).groupby("subject_id", sort=False):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group[label_col].to_numpy(dtype=np.float32)
        h = group["hour"].to_numpy(dtype=np.int32)
        ends = list(range(0, len(group), max(1, int(stride))))
        if ends and ends[-1] != len(group) - 1:
            ends.append(len(group) - 1)
        if max_windows_per_patient and len(ends) > max_windows_per_patient:
            pick = np.linspace(0, len(ends) - 1, max_windows_per_patient).round().astype(int)
            ends = [ends[i] for i in sorted(set(pick))]
        for end in ends:
            start = max(0, end - seq_len + 1)
            seq = values[start:end + 1]
            if len(seq) < seq_len:
                # Repeat the earliest available observation for left padding; this
                # avoids injecting an artificial all-zero clinical measurement.
                pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
                seq = np.concatenate([pad, seq], axis=0)
            xs.append(seq)
            ys.append(labels[end])
            ids.append(str(pid))
            hours.append(int(h[end]))
    if not xs:
        raise ValueError("No sequence windows could be created")
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(ids), np.asarray(hours)
