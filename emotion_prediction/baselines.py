# baselines.py
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Data Loading Functions ----------
def load_csv_dataset(train_csv_path: str, test_csv_path: str, toggle_face: bool = False):
    """Load dataset from CSV files and gaze features from .npy files."""
    
    print(f"[data] Loading training data from: {train_csv_path}")
    print(f"[data] Loading testing data from: {test_csv_path}")

    train_df, test_df = (pd.read_csv(train_csv_path), pd.read_csv(test_csv_path))
    
    # Load gaze features from .npy files
    train_csv_path = Path(train_csv_path)
    test_csv_path = Path(test_csv_path)
    
    train_gaze_path = train_csv_path.parent / "train_gaze.npy"
    test_gaze_path = test_csv_path.parent / "test_gaze.npy"
    
    print(f"[data] Loading training gaze features from: {train_gaze_path}")
    print(f"[data] Loading testing gaze features from: {test_gaze_path}")
    
    train_gaze = np.load(train_gaze_path)
    test_gaze = np.load(test_gaze_path)
    
    face_cols = [c for c in train_df.columns if toggle_face and c.startswith('face_feature_')]
    label_cols = ['valence', 'arousal', 'dominance']
    
    # Drop NaN labels in one line
    train_df = train_df.dropna(subset=label_cols)
    test_df = test_df.dropna(subset=label_cols)

    # Extract as NumPy directly
    def extract(df, gaze_data, use_face):
        X = gaze_data  # Gaze features are already loaded from .npy
        Y = df[label_cols].to_numpy(dtype=np.float32)
        Z = df[face_cols].to_numpy(dtype=np.float32) if use_face else None
        return X, Y, Z
    
    X_train, Y_train, Z_train = extract(train_df, train_gaze, toggle_face)
    X_test, Y_test, Z_test = extract(test_df, test_gaze, toggle_face)
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test


def _infer_window_size_from_features(X: np.ndarray) -> int:
    """
    Infer window size from a 2D feature matrix shaped [N, features],
    where features = window_size * 6 (two eyes, 3 axes each).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array shaped [N, features], got {X.shape}")
    features = X.shape[1]
    if features % 6 != 0:
        raise ValueError(f"Feature dimension {features} is not divisible by 6; cannot infer window size")
    return features // 6


def reshape_features_for_cnn(X: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
    """
    Reshape flattened features back to window format for CNN processing
    
    Args:
        X: Features array of shape [N, features] where features = 6 * window_size
    
    Returns:
        Reshaped array of shape [N, 6, window_size] for CNN processing
    """
    if window_size is None:
        window_size = _infer_window_size_from_features(X)
    if X.shape[1] != window_size * 6:
        raise ValueError(f"Expected {window_size * 6} features, got {X.shape[1]}")
    
    # Reshape to [N, 6, window_size] for CNN
    X_reshaped = X.reshape(-1, 6, window_size)
    return X_reshaped


# ---------- Metrics ----------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return per-dim and average MSE/MAE/R2 and Pearson r."""
    assert y_true.shape == y_pred.shape
    eps = 1e-12
    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    mae = np.abs(y_true - y_pred).mean(axis=0)
    # r2 per dim
    var = y_true.var(axis=0) + eps
    r2 = 1.0 - ((y_true - y_pred) ** 2).mean(axis=0) / var
    # Pearson r per dim
    y0 = y_true - y_true.mean(axis=0, keepdims=True)
    p0 = y_pred - y_pred.mean(axis=0, keepdims=True)
    pear = (y0 * p0).sum(axis=0) / (np.sqrt((y0 ** 2).sum(axis=0) * (p0 ** 2).sum(axis=0)) + eps)
    out = {
        "mse": mse, "mae": mae, "r2": r2, "pearson": pear,
        "mse_avg": float(mse.mean()), "mae_avg": float(mae.mean()),
        "r2_avg": float(r2.mean()), "pearson_avg": float(pear.mean()),
    }
    return out


# ============================================================
# =============== 1) CNN (PyTorch) Baseline ==================
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    """
    Holds features and labels for CNN training.
    Features are already flattened and will be reshaped to [6, window_size].
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, window_size: Optional[int] = None):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.window_size = _infer_window_size_from_features(X) if window_size is None else window_size
        if self.X.shape[1] != self.window_size * 6:
            raise ValueError(
                f"WindowDataset: feature dimension {self.X.shape[1]} does not match expected {self.window_size * 6}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # [features] where features = window_size * 6
        y = self.Y[idx]  # [3]
        
        # Reshape to [6, window_size] for CNN
        x_reshaped = x.reshape(6, self.window_size)
        
        return x_reshaped.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def pad_collate(batch):
    """
    Since all windows within a dataset are the same size, no padding is needed.
    Returns:
      x: [B, 6, window_size]
      y: [B, 3]
    """
    xs, ys = zip(*batch)
    x_batch = np.stack(xs, axis=0)  # [B, 6, window_size]
    y_batch = np.stack(ys, axis=0)  # [B, 3]
    
    return (
        torch.from_numpy(x_batch),
        torch.from_numpy(y_batch),
    )


class TemporalCNN(nn.Module):
    def __init__(self, in_ch=6, hidden=64, depth=3, kernel=5, dropout=0.1, out_dim=3):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=kernel, padding=kernel//2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
            ]
            ch = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: [B, 6, T]
        h = self.body(x)                            # [B, H, T]
        h_pool = h.mean(dim=2)                      # [B, H] - Global average pooling
        y = self.head(h_pool)                       # [B, 3]
        return y


@torch.no_grad()
def _evaluate_cnn(model, loader, device="cpu"):
    ys, ps = [], []
    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    if not ys:
        return {}
    y = np.concatenate(ys, axis=0); p = np.concatenate(ps, axis=0)
    return _metrics(y, p)


def train_cnn_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    window_size: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if window_size is None:
        window_size = _infer_window_size_from_features(X_train)
    if X_test.shape[1] != window_size * 6:
        raise ValueError(
            f"X_test feature dimension {X_test.shape[1]} does not match inferred window size {window_size} * 6"
        )
    # Split a small validation set from train windows (10%)
    n = len(X_train)
    idx = np.arange(n)
    rng = np.random.default_rng(123)
    rng.shuffle(idx)
    val_n = max(1, int(0.1 * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    X_tr = X_train[tr_idx]; Y_tr = Y_train[tr_idx]
    X_va = X_train[val_idx]; Y_va = Y_train[val_idx]

    ds_tr = WindowDataset(X_tr, Y_tr, window_size=window_size)
    ds_va = WindowDataset(X_va, Y_va, window_size=window_size)
    ds_te = WindowDataset(X_test, Y_test, window_size=window_size)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)

    model = TemporalCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_va = math.inf
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        loss_acc = 0.0; n_seen = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_acc += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        sched.step()
        va_metrics = _evaluate_cnn(model, dl_va, device=device)
        va_mse = va_metrics.get("mse_avg", float("inf"))
        print(f"[CNN] epoch {ep:02d} | train loss {loss_acc/max(n_seen,1):.4f} | val MSE {va_mse:.4f} | val r {va_metrics.get('pearson_avg', 0):.3f}")
        if va_mse < best_va:
            best_va = va_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_metrics = _evaluate_cnn(model, dl_te, device=device)
    print(f"[CNN] TEST: MSE_avg={te_metrics.get('mse_avg', float('nan')):.4f}, "
          f"MAE_avg={te_metrics.get('mae_avg', float('nan')):.4f}, "
          f"R2_avg={te_metrics.get('r2_avg', float('nan')):.3f}, "
          f"r_avg={te_metrics.get('pearson_avg', float('nan')):.3f}")
    return model, te_metrics


# ============================================================
# ======== 2) Statistical (feature) Regressor Baseline ======
# ============================================================

def _featureize_window(x_win: np.ndarray, window_size: Optional[int] = None, face_feats: Optional[np.ndarray] = None) -> np.ndarray:
    """
    x_win: [window_size * 6] flattened features representing a 5s window.
    We'll reshape and extract statistical features.
    
    Features (per eye):
      - mean/std of x,y,z (6)
      - velocity magnitude: mean/std/median/p95 (4)
      - acceleration magnitude: mean/std (2)
    Cross-eye:
      - corr(x), corr(y), corr(z) between eyes (3)  (nan->0)
    Global:
      - T (frame count), approx fps (=T/win_sec) (2)

    Total dims = 2* (6 + 4 + 2) + 3 + 2 = 2*12 + 3 + 2 = 29
    """
    # Reshape to [window_size, 6]
    if window_size is None:
        if x_win.ndim != 1:
            raise ValueError("x_win must be a 1D array when inferring window size")
        if x_win.size % 6 != 0:
            raise ValueError(f"Cannot infer window size from length {x_win.size}; not divisible by 6")
        window_size = x_win.size // 6
    x_reshaped = x_win.reshape(window_size, 6)
    
    # split eyes
    g0 = x_reshaped[:, 0:3]   # [T,3]
    g1 = x_reshaped[:, 3:6]   # [T,3]

    # means/stds
    def mean_std(mat):
        return np.concatenate([mat.mean(axis=0), mat.std(axis=0)], axis=0)  # 6

    # velocities/accelerations (approximate dt)
    dt = max(1e-6, 5.0 / float(window_size))  # seconds/sample (since 5s window)
    v0 = np.diff(g0, axis=0) / dt                       # [T-1,3]
    v1 = np.diff(g1, axis=0) / dt
    a0 = np.diff(v0, axis=0) / dt if v0.shape[0] > 1 else np.zeros_like(v0)
    a1 = np.diff(v1, axis=0) / dt if v1.shape[0] > 1 else np.zeros_like(v1)

    def mag_stats(v):
        mag = np.linalg.norm(v, axis=1)  # [n]
        if mag.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.median(mag)),
            float(np.percentile(mag, 95)),
        ], dtype=np.float32)

    def accel_stats(a):
        mag = np.linalg.norm(a, axis=1)
        if mag.size == 0:
            return np.zeros(2, dtype=np.float32)
        return np.array([float(np.mean(mag)), float(np.std(mag))], dtype=np.float32)

    # cross-eye correlations per axis
    def safe_corr(u, v):
        u = u - u.mean()
        v = v - v.mean()
        su = float(np.sqrt((u * u).sum())) + 1e-12
        sv = float(np.sqrt((v * v).sum())) + 1e-12
        return float((u * v).sum() / (su * sv))

    cross = np.array([
        safe_corr(g0[:, 0], g1[:, 0]),
        safe_corr(g0[:, 1], g1[:, 1]),
        safe_corr(g0[:, 2], g1[:, 2]),
    ], dtype=np.float32)

    feats = np.concatenate([
        mean_std(g0), mag_stats(v0), accel_stats(a0),   # 6 + 4 + 2
        mean_std(g1), mag_stats(v1), accel_stats(a1),   # 6 + 4 + 2
        cross,                                          # 3
        np.array([float(window_size), float(window_size) / 5.0], dtype=np.float32)  # 2
    ], axis=0)
    assert feats.shape[0] == 29

    # If face features are provided, concatenate them
    if face_feats is not None:
        feats = np.concatenate([feats, face_feats.astype(np.float32)], axis=0)

    return feats.astype(np.float32)


# --- Feature Engineering (Vectorized) ---
def _build_feature_matrix(X, window_size: Optional[int] = None, Z=None):
    N = len(X)
    if N == 0: 
        return np.zeros((0, 29 if Z is None else 29 + Z.shape[1]), dtype=np.float32)

    if window_size is None:
        window_size = _infer_window_size_from_features(X)
    Xr = X.reshape(N, window_size, 6)
    g0, g1 = Xr[:, :, :3], Xr[:, :, 3:]

    def stats(g):
        mean, std = g.mean(1), g.std(1)
        dt = 5.0 / window_size
        v = np.diff(g, axis=1) / dt
        a = np.diff(v, axis=1) / dt
        mag_v = np.linalg.norm(v, axis=2)
        mag_a = np.linalg.norm(a, axis=2)
        feats = np.concatenate([
            mean, std,
            mag_v.mean(1, keepdims=True), mag_v.std(1, keepdims=True),
            np.median(mag_v, 1, keepdims=True), np.percentile(mag_v, 95, axis=1, keepdims=True),
            mag_a.mean(1, keepdims=True), mag_a.std(1, keepdims=True)
        ], axis=1)
        return feats

    feats0, feats1 = stats(g0), stats(g1)

    # fixed here ↓
    cross = np.stack([
        [np.corrcoef(g0[:, i], g1[:, i])[0, 1] for i in range(3)]
        for g0, g1 in zip(g0, g1)
    ])

    global_feats = np.tile([window_size, window_size / 5.0], (N, 1))
    Phi = np.concatenate([feats0, feats1, cross, global_feats], axis=1)

    if Z is not None:
        Phi = np.concatenate([Phi, Z.astype(np.float32)], axis=1)

    return Phi.astype(np.float32)


def train_stats_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Z_train: Optional[np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Z_test: Optional[np.ndarray],
    alpha: float = 1.0,
    window_size: Optional[int] = None,
):
    """Multi-output Ridge regression with a small, robust feature set."""
    if window_size is None:
        window_size = _infer_window_size_from_features(X_train)
    if X_test.shape[1] != window_size * 6:
        raise ValueError(
            f"X_test feature dimension {X_test.shape[1]} does not match inferred window size {window_size} * 6"
        )
    # Prefer sklearn if available, otherwise fallback to closed-form ridge
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              Ridge(alpha=alpha, fit_intercept=True, random_state=0))
        Phi_tr = _build_feature_matrix(X_train, window_size, Z_train)
        Phi_te = _build_feature_matrix(X_test, window_size, Z_test)
        model.fit(Phi_tr, Y_train)
        Y_pred = model.predict(Phi_te)
        m = _metrics(Y_test, Y_pred)
        print(f"[STATS] TEST: MSE_avg={m['mse_avg']:.4f}, MAE_avg={m['mae_avg']:.4f}, "
              f"R2_avg={m['r2_avg']:.3f}, r_avg={m['pearson_avg']:.3f}")
        return model, m
    except Exception as e:
        print(f"[STATS] sklearn unavailable or failed ({e}); falling back to NumPy ridge.")

        # simple ridge fit: (Phi^T Phi + alpha I) W = Phi^T Y
        Phi_tr = _build_feature_matrix(X_train, window_size, Z_train).astype(np.float64)
        Phi_te = _build_feature_matrix(X_test, window_size, Z_test).astype(np.float64)
        # standardize Phi
        mu = Phi_tr.mean(axis=0, keepdims=True)
        sd = Phi_tr.std(axis=0, keepdims=True) + 1e-8
        Phi_trn = (Phi_tr - mu) / sd
        Phi_ten = (Phi_te - mu) / sd

        I = np.eye(Phi_trn.shape[1], dtype=np.float64)
        A = Phi_trn.T @ Phi_trn + alpha * I
        B = Phi_trn.T @ Y_train.astype(np.float64)
        W = np.linalg.solve(A, B)  # [D,3]
        Y_pred = Phi_ten @ W
        m = _metrics(Y_test, Y_pred.astype(np.float32))
        print(f"[STATS] TEST: MSE_avg={m['mse_avg']:.4f}, MAE_avg={m['mae_avg']:.4f}, "
              f"R2_avg={m['r2_avg']:.3f}, r_avg={m['pearson_avg']:.3f}")
        return (mu, sd, W), m


# ============================================================
# ==================== Script Entrypoint =====================
# ============================================================

if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Train baseline models on eye gaze emotion dataset")
    parser.add_argument("--data_folder", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for CNN (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CNN training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for CNN (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for CNN (default: 1e-4)")
    
    args = parser.parse_args()
    
    # 0) Load dataset from CSV files
    train_csv = args.data_folder / "train.csv"
    test_csv = args.data_folder / "test.csv"
    X_train, Y_train, Z_train, X_test, Y_test, Z_test = load_csv_dataset(train_csv, test_csv, toggle_face=True)

    print(f"[data] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("[data] No samples found. Check your CSV files.")

    # Create results directory structure
    results_dir = Path("results") / args.data_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results storage
    results_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1) Train STATISTICAL baseline with face 
    print("\n" + "="*50)
    print("TRAINING STATISTICAL BASELINE WITH FACE FEATURES")
    print("="*50)
    stats_face_model, stats_face_metrics = train_stats_baseline(X_train, Y_train, Z_train, X_test, Y_test, Z_test, alpha=2.0)
    
    # Store results
    results_data.append({
        'model': 'STATISTICAL_WITH_FACE',
        'timestamp': timestamp,
        'mse_avg': float(stats_face_metrics['mse_avg']),
        'mae_avg': float(stats_face_metrics['mae_avg']),
        'r2_avg': float(stats_face_metrics['r2_avg']),
        'pearson_avg': float(stats_face_metrics['pearson_avg']),
        'mse_valence': float(stats_face_metrics['mse'][0]),
        'mse_arousal': float(stats_face_metrics['mse'][1]),
        'mse_dominance': float(stats_face_metrics['mse'][2]),
        'mae_valence': float(stats_face_metrics['mae'][0]),
        'mae_arousal': float(stats_face_metrics['mae'][1]),
        'mae_dominance': float(stats_face_metrics['mae'][2]),
        'r2_valence': float(stats_face_metrics['r2'][0]),
        'r2_arousal': float(stats_face_metrics['r2'][1]),
        'r2_dominance': float(stats_face_metrics['r2'][2]),
        'pearson_valence': float(stats_face_metrics['pearson'][0]),
        'pearson_arousal': float(stats_face_metrics['pearson'][1]),
        'pearson_dominance': float(stats_face_metrics['pearson'][2]),
        'epochs': None,
        'batch_size': None,
        'lr': None,
        'weight_decay': None
    })

    # 2) Train STATISTICAL baseline without face
    print("\n" + "="*50)
    print("TRAINING STATISTICAL BASELINE")
    print("="*50)
    stats_model, stats_metrics = train_stats_baseline(X_train, Y_train, None, X_test, Y_test, None, alpha=2.0)
    
    # Store results
    results_data.append({
        'model': 'STATISTICAL_NO_FACE',
        'timestamp': timestamp,
        'mse_avg': float(stats_metrics['mse_avg']),
        'mae_avg': float(stats_metrics['mae_avg']),
        'r2_avg': float(stats_metrics['r2_avg']),
        'pearson_avg': float(stats_metrics['pearson_avg']),
        'mse_valence': float(stats_metrics['mse'][0]),
        'mse_arousal': float(stats_metrics['mse'][1]),
        'mse_dominance': float(stats_metrics['mse'][2]),
        'mae_valence': float(stats_metrics['mae'][0]),
        'mae_arousal': float(stats_metrics['mae'][1]),
        'mae_dominance': float(stats_metrics['mae'][2]),
        'r2_valence': float(stats_metrics['r2'][0]),
        'r2_arousal': float(stats_metrics['r2'][1]),
        'r2_dominance': float(stats_metrics['r2'][2]),
        'pearson_valence': float(stats_metrics['pearson'][0]),
        'pearson_arousal': float(stats_metrics['pearson'][1]),
        'pearson_dominance': float(stats_metrics['pearson'][2]),
        'epochs': None,
        'batch_size': None,
        'lr': None,
        'weight_decay': None
    })

    # 3) Train CNN baseline
    print("\n" + "="*50)
    print("TRAINING CNN BASELINE")
    print("="*50)
    cnn_model, cnn_metrics = train_cnn_baseline(
        X_train, Y_train, X_test, Y_test,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Store results
    results_data.append({
        'model': 'CNN',
        'timestamp': timestamp,
        'mse_avg': float(cnn_metrics['mse_avg']),
        'mae_avg': float(cnn_metrics['mae_avg']),
        'r2_avg': float(cnn_metrics['r2_avg']),
        'pearson_avg': float(cnn_metrics['pearson_avg']),
        'mse_valence': float(cnn_metrics['mse'][0]),
        'mse_arousal': float(cnn_metrics['mse'][1]),
        'mse_dominance': float(cnn_metrics['mse'][2]),
        'mae_valence': float(cnn_metrics['mae'][0]),
        'mae_arousal': float(cnn_metrics['mae'][1]),
        'mae_dominance': float(cnn_metrics['mae'][2]),
        'r2_valence': float(cnn_metrics['r2'][0]),
        'r2_arousal': float(cnn_metrics['r2'][1]),
        'r2_dominance': float(cnn_metrics['r2'][2]),
        'pearson_valence': float(cnn_metrics['pearson'][0]),
        'pearson_arousal': float(cnn_metrics['pearson'][1]),
        'pearson_dominance': float(cnn_metrics['pearson'][2]),
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'lr': float(args.lr),
        'weight_decay': float(args.weight_decay)
    })
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    # Save results to JSON
    import json
    
    # Create combined results structure
    combined_results = {
        'timestamp': timestamp,
        'data_folder': str(args.data_folder),
        'detailed_results': results_data,
        'summary': []
    }
    
    # Add summary data
    for result in results_data:
        combined_results['summary'].append({
            'model': result['model'],
            'mse_avg': result['mse_avg'],  # Already converted to float above
            'mae_avg': result['mae_avg'],  # Already converted to float above
            'r2_avg': result['r2_avg'],    # Already converted to float above
            'pearson_avg': result['pearson_avg']  # Already converted to float above
        })
    
    # Save to JSON file
    results_json_path = results_dir / "results.json"
    with open(results_json_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n[results] Results saved to: {results_json_path}")


"""
python baselines.py --data_folder data/5.0_seconds/seed_123
python baselines.py --data_folder data/5.0_seconds/seed_234
python baselines.py --data_folder data/5.0_seconds/seed_345
python baselines.py --data_folder data/5.0_seconds/seed_456
python baselines.py --data_folder data/5.0_seconds/seed_567

python baselines.py --data_folder data/2.0_seconds/seed_123
python baselines.py --data_folder data/2.0_seconds/seed_234
python baselines.py --data_folder data/2.0_seconds/seed_345
python baselines.py --data_folder data/2.0_seconds/seed_456
python baselines.py --data_folder data/2.0_seconds/seed_567

python baselines.py --data_folder data/10.0_seconds/seed_123
python baselines.py --data_folder data/10.0_seconds/seed_234
python baselines.py --data_folder data/10.0_seconds/seed_345
python baselines.py --data_folder data/10.0_seconds/seed_456
python baselines.py --data_folder data/10.0_seconds/seed_567

python baselines.py --data_folder data_normalized/5.0_seconds/seed_123
python baselines.py --data_folder data_normalized/5.0_seconds/seed_234
python baselines.py --data_folder data_normalized/5.0_seconds/seed_345
python baselines.py --data_folder data_normalized/5.0_seconds/seed_456
python baselines.py --data_folder data_normalized/5.0_seconds/seed_567

python baselines.py --data_folder data_normalized/2.0_seconds/seed_123
python baselines.py --data_folder data_normalized/2.0_seconds/seed_234
python baselines.py --data_folder data_normalized/2.0_seconds/seed_345
python baselines.py --data_folder data_normalized/2.0_seconds/seed_456
python baselines.py --data_folder data_normalized/2.0_seconds/seed_567

python baselines.py --data_folder data_normalized/10.0_seconds/seed_123
python baselines.py --data_folder data_normalized/10.0_seconds/seed_234
python baselines.py --data_folder data_normalized/10.0_seconds/seed_345
python baselines.py --data_folder data_normalized/10.0_seconds/seed_456
python baselines.py --data_folder data_normalized/10.0_seconds/seed_567

"""