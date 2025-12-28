from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def quick_train_autoencoder(
    model: torch.nn.Module,
    X_train: np.ndarray,
    epochs: int = 12,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> None:
    model.to(device)
    model.train()

    ds = TensorDataset(torch.from_numpy(X_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

def reconstruction_error_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    model.to(device)
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    errs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            recon = model(xb)
            e = torch.mean((recon - xb) ** 2, dim=(1, 2))
            errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)

def expand_window_scores_to_timeline(window_scores: np.ndarray, T: int, window: int) -> np.ndarray:

    timeline = np.zeros(T, dtype="float32")
    counts = np.zeros(T, dtype="float32")
    for i, s in enumerate(window_scores):
        start = i
        end = i + window
        timeline[start:end] += float(s)
        counts[start:end] += 1.0
    counts = np.maximum(counts, 1.0)
    return timeline / counts

def pick_threshold(train_window_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(train_window_scores, percentile))

def risk_label(anomaly_mask: np.ndarray) -> str:

    cnt = int(np.sum(anomaly_mask))
    if cnt < 3:
        return "Low"

    run = 0
    max_run = 0
    for v in anomaly_mask.astype(int):
        if v == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run >= 5:
        return "High"
    return "Medium"
