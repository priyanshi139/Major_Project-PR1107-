"""Beat segmentation + features. Classes follow AAMI grouping of MIT-BIH annotation symbols.
Beats with other symbols (paced, fusion of paced, unclassifiable, non-beat marks) are excluded."""
import numpy as np

AAMI_MAP = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
            "A": "S", "a": "S", "J": "S", "S": "S", "V": "V", "E": "V", "F": "F"}
CLASSES = ["N", "S", "V", "F"]
PRE, POST = 90, 110     # samples around the annotated R peak at 360 Hz (about 0.25 s before, 0.31 s after)


def extract_beats(signal, samples, symbols, fs=360, pre=PRE, post=POST) -> dict:
    sig = np.asarray(signal, float)
    sig = (sig - np.median(sig)) / (np.std(sig) + 1e-9)               # per-record normalisation
    samples = np.asarray(samples, int)
    rr_all = np.diff(samples) / fs
    med_rr = float(np.median(rr_all)) if len(rr_all) else 1.0
    waves, rr, labels, kept = [], [], [], []
    for i, (s, sym) in enumerate(zip(samples, symbols)):
        cls = AAMI_MAP.get(sym)
        if cls is None or s - pre < 0 or s + post > len(sig):
            continue
        pre_rr = (samples[i] - samples[i - 1]) / fs if i > 0 else med_rr
        post_rr = (samples[i + 1] - samples[i]) / fs if i < len(samples) - 1 else med_rr
        lo = max(0, i - 10)
        local = float(np.mean(np.diff(samples[lo:i + 1]) / fs)) if i - lo >= 1 else med_rr
        waves.append(sig[s - pre:s + post]); rr.append([pre_rr, post_rr, local, pre_rr / (local + 1e-9)])
        labels.append(cls); kept.append(s)
    n = len(waves)
    return {"waves": np.array(waves).reshape(n, pre + post), "rr": np.array(rr).reshape(n, 4),
            "labels": np.array(labels), "samples": np.array(kept, int)}


def feature_matrix(waves, rr) -> np.ndarray:
    w = np.asarray(waves)
    stats = np.c_[w.min(1), w.max(1), w.max(1) - w.min(1), w.std(1), w.argmax(1), w.argmin(1)] if len(w) else np.zeros((0, 6))
    return np.hstack([w[:, ::2], np.asarray(rr), stats])
