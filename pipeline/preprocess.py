from __future__ import annotations
import pandas as pd
import numpy as np

def load_timeseries_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    if "value" not in df.columns:
        raise ValueError("CSV must contain a 'value' column.")
    if "timestamp" not in df.columns:
        # allow index-based sequence
        df = df.copy()
        df["timestamp"] = np.arange(len(df))
    return df[["timestamp", "value"]].copy()

def normalize_zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = values.astype("float32")
    mean = float(np.mean(values))
    std = float(np.std(values) + 1e-8)
    return (values - mean) / std, mean, std

def window_sequence(values: np.ndarray, window: int) -> np.ndarray:
    """
    Create overlapping windows: (N, window, 1)
    """
    if len(values) < window:
        raise ValueError(f"Need at least {window} points, got {len(values)}.")
    X = []
    for i in range(0, len(values) - window + 1):
        X.append(values[i:i+window])
    X = np.stack(X, axis=0).astype("float32")  # (N, window)
    return X[..., None]  # (N, window, 1)
