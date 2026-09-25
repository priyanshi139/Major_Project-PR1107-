"""
HealthTwin — Digital Twin State (Sequence Formatting) + PyTorch Dataset
Converts per-patient hourly records into fixed-length padded sequences
for Transformer model input.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def build_sequences(df, feature_cols, label_col='SepsisLabel',
                     patient_col='PatientID', max_len=60):
    """
    Convert a flat dataframe (one row per patient-hour) into
    fixed-length padded sequences per patient.

    max_len=60 covers ~95th percentile of ICU stay length in this
    dataset (median stay ~38 hours); longer stays are truncated.
    """
    grouped = df.groupby(patient_col)
    n_patients = grouped.ngroups
    n_features = len(feature_cols)

    sequences = np.zeros((n_patients, max_len, n_features), dtype=np.float32)
    masks = np.zeros((n_patients, max_len), dtype=np.float32)
    labels = np.zeros((n_patients, max_len), dtype=np.float32)
    patient_ids = []

    for i, (pid, group) in enumerate(grouped):
        group = group.sort_values('ICULOS')
        feats = group[feature_cols].values.astype(np.float32)
        label = group[label_col].values.astype(np.float32)

        seq_len = min(len(feats), max_len)
        sequences[i, :seq_len] = feats[:seq_len]
        labels[i, :seq_len] = label[:seq_len]
        masks[i, :seq_len] = 1
        patient_ids.append(pid)

    return sequences, masks, labels, patient_ids


class SepsisDataset(Dataset):
    """PyTorch Dataset wrapping padded sequences, masks, and labels."""

    def __init__(self, sequences, masks, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.labels[idx]


if __name__ == "__main__":
    train_df = pd.read_csv("dataset/healthtwin_sepsis_data/preprocessed/train_preprocessed.csv")
    test_df = pd.read_csv("dataset/healthtwin_sepsis_data/preprocessed/test_preprocessed.csv")

    feature_cols = [c for c in train_df.columns if c not in ['PatientID', 'SepsisLabel']]

    train_seq, train_mask, train_labels, train_pids = build_sequences(train_df, feature_cols, max_len=60)
    test_seq, test_mask, test_labels, test_pids = build_sequences(test_df, feature_cols, max_len=60)

    print("Train sequences shape:", train_seq.shape)
    print("Test sequences shape:", test_seq.shape)

    train_dataset = SepsisDataset(train_seq, train_mask, train_labels)
    test_dataset = SepsisDataset(test_seq, test_mask, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    batch_seq, batch_mask, batch_label = next(iter(train_loader))
    print("Batch sequence shape:", batch_seq.shape)
