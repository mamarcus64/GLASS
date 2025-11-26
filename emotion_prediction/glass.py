import torch
import numpy as np
from pathlib import Path
import sys

# Handle imports for both standalone and package use
try:
    from .baselines import load_csv_dataset, _metrics
except ImportError:
    from baselines import load_csv_dataset, _metrics

# Add self_supervised_gaze to path for model imports
_glass_root = Path(__file__).parent.parent
if str(_glass_root / "self_supervised_gaze") not in sys.path:
    sys.path.insert(0, str(_glass_root / "self_supervised_gaze"))

from models.patch_transformer import PatchSeq2Seq

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x


def get_embeddings(ckpt_loc: str, eye_movements: list[np.ndarray], batch_size: int = 256) -> tuple[np.ndarray, int]:
    """
    Load a trained PatchSeq2Seq model checkpoint and extract the FULL ENCODER TOKEN SEQUENCES
    for each input sequence of eye movements.

    Args:
        ckpt_loc: Path to a training checkpoint saved by train.py (must contain "model" and "config").
        eye_movements: list of arrays, each of shape (T, D), e.g., (900, 6) or (Tp, 6).
                       If T != Tp from the checkpoint config, sequences are trimmed/padded to length Tp.
        batch_size: mini-batch size for embedding extraction.

    Returns:
        np.ndarray of shape (N, N_e, d_model) where N_e = Tp // patch_size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load checkpoint and instantiate model ----
    state = torch.load(ckpt_loc, map_location="cpu")
    cfg = state["config"]
    model = PatchSeq2Seq(cfg).to(device)

    sd = state["model"]
    # handle DataParallel checkpoints if needed
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)

    model.eval()

    # ---- config-driven lengths/dims ----
    Tp = int(cfg["data"]["past_len"])
    D  = int(cfg["model"]["D_in"])
    P  = int(cfg["model"]["patch_size"])
    d_model = int(cfg["model"]["d_model"])
    assert Tp % P == 0, f"past_len ({Tp}) must be divisible by patch_size ({P})."
    N_e = Tp // P
    
    # sanity on input dims
    for i, arr in enumerate(eye_movements):
        if arr.shape[1] != D:
            raise ValueError(f"Input #{i} has D={arr.shape[1]} (expected {D}).")

    # ---- helper: normalize sequence length to Tp ----
    def _fix_length(x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        if T >= Tp:
            x = x[-Tp:, :]  # use the most recent Tp frames
        else:
            pad = np.tile(x[-1:, :], (Tp - T, 1))  # repeat last frame
            x = np.concatenate([x, pad], axis=0)
        return x.astype(np.float32, copy=False)

    # ---- batching ----
    N = len(eye_movements)
    out = np.empty((N, N_e, d_model), dtype=np.float32)

    with torch.no_grad():
        for idx in tqdm(range(0, N, batch_size), total=(N + batch_size - 1) // batch_size, desc="Embeddings"):
            j = min(idx + batch_size, N)
            batch_np = np.stack([_fix_length(eye_movements[k]) for k in range(idx, j)], axis=0)  # (B, Tp, D)
            batch = torch.from_numpy(batch_np).to(device, non_blocking=True)                     # (B, Tp, D)

            # encoder token sequences
            enc_tokens = model.patch_embed_enc(batch)   # (B, N_e, d_model)
            enc_out = model.encoder(enc_tokens)         # (B, N_e, d_model)

            out[idx:j, :, :] = enc_out.detach().cpu().numpy()

    return out, P

def unroll_eye_data(eye_data: np.ndarray):
    """
    Unroll eye data into a list of arrays, each of shape (T, D), where T is the number of frames and D is the number of dimensions.
    """
    assert eye_data.shape[1] % 6 == 0, "Eye data must have 6 columns"
    num_frames = eye_data.shape[1] // 6
    data = []
    for i in range(eye_data.shape[0]):
        data.append(eye_data[i].reshape(num_frames, 6))
    return data


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert len(X) == len(Y)
        self.X = X.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set for any environment variables
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def _compute_deltas(Z: torch.Tensor):
    dZ = torch.zeros_like(Z)
    dZ[:, 1:, :] = Z[:, 1:, :] - Z[:, :-1, :]
    ddZ = torch.zeros_like(Z)
    ddZ[:, 1:, :] = dZ[:, 1:, :] - dZ[:, :-1, :]
    return dZ, ddZ


class RichPoolMLP(nn.Module):
    def __init__(self, d: int, out_dim: int = 3, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        # Separate learnable queries for Z, dZ, ddZ
        self.q_z = nn.Parameter(torch.randn(d))
        self.q_dz = nn.Parameter(torch.randn(d))
        self.q_ddz = nn.Parameter(torch.randn(d))
        # Each stream contributes [mean, std, max, attnpool] -> 4*d dims; total 3 streams
        in_dim = 12 * d
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def _pool_stream(self, Z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Z: [B,T,d]
        mean = Z.mean(dim=1)
        std = Z.std(dim=1, unbiased=False)
        maxv, _ = Z.max(dim=1)
        q_exp = q[None, None, :].expand(Z.size(0), 1, Z.size(2))
        scores = torch.einsum('btd,bsd->bts', Z, q_exp) / (Z.size(2) ** 0.5)
        w = torch.softmax(scores.squeeze(-1), dim=1)
        attn = torch.einsum('bt,btd->bd', w, Z)
        return torch.cat([mean, std, maxv, attn], dim=-1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        dZ, ddZ = _compute_deltas(Z)
        pooled_z = self._pool_stream(Z, self.q_z)
        pooled_dz = self._pool_stream(dZ, self.q_dz)
        pooled_ddz = self._pool_stream(ddZ, self.q_ddz)
        feat = torch.cat([pooled_z, pooled_dz, pooled_ddz], dim=-1)
        y = self.mlp(feat)
        return torch.clamp(y, 0.0, 1.0)


class ChunkedRichPoolMLP(nn.Module):
    def __init__(self, d: int, out_dim: int = 3, hidden: int = 512, dropout: float = 0.3,
                 chunk_splits: tuple[int, ...] = (1, 2, 4), tokens_per_second: int | None = None,
                 head_arch: str = "mlp",
                 tcn_channels: int = 128, tcn_blocks: int = 2, tcn_dilations: tuple[int, ...] = (1, 2, 4),
                 gru_hidden: int = 128, gru_layers: int = 1,
                 tf_d_model: int = 128, tf_heads: int = 4, tf_layers: int = 1):
        super().__init__()
        # Interpret chunk_splits as durations in seconds.
        self.chunk_seconds = tuple(int(s) for s in chunk_splits)
        self.tokens_per_second = 1 if tokens_per_second is None else int(max(1, tokens_per_second))
        # queries per stream
        self.q_z = nn.Parameter(torch.randn(d))
        self.q_dz = nn.Parameter(torch.randn(d))
        self.q_ddz = nn.Parameter(torch.randn(d))
        self.head_arch = head_arch.lower()
        self.dropout = dropout
        # Heads will be constructed lazily on first forward when chunk feature dim is known
        self._mlp_head = None
        self._tcn = None
        self._tcn_proj = None
        self._tcn_q = None
        self._gru = None
        self._gru_q = None
        self._tf_in = None
        self._tf_enc = None
        self._tf_q = None
        self._final_mlp = None
        self._cfg = {
            'hidden': hidden,
            'tcn_channels': tcn_channels,
            'tcn_blocks': tcn_blocks,
            'tcn_dilations': tcn_dilations,
            'gru_hidden': gru_hidden,
            'gru_layers': gru_layers,
            'tf_d_model': tf_d_model,
            'tf_heads': tf_heads,
            'tf_layers': tf_layers,
            'out_dim': out_dim,
        }

    def _per_chunk_features(self, Z: torch.Tensor, q: torch.Tensor) -> list[torch.Tensor]:
        B, T, d = Z.shape
        feats = []
        for sec in self.chunk_seconds:
            seg = max(1, int(round(self.tokens_per_second * sec)))
            for start in range(0, T, seg):
                end = min(T, start + seg)
                Zi = Z[:, start:end, :]
                mean = Zi.mean(dim=1)
                std = Zi.std(dim=1, unbiased=False)
                maxv, _ = Zi.max(dim=1)
                q_exp = q[None, None, :].expand(Zi.size(0), 1, d)
                scores = torch.einsum('btd,bsd->bts', Zi, q_exp) / (d ** 0.5)
                w = torch.softmax(scores.squeeze(-1), dim=1)
                attn = torch.einsum('bt,btd->bd', w, Zi)
                feats.append(torch.cat([mean, std, maxv, attn], dim=-1))
        return feats

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, d = Z.shape
        dZ, ddZ = _compute_deltas(Z)
        # Build per-chunk features for each stream (same chunking)
        feats_z = self._per_chunk_features(Z, self.q_z)
        feats_dz = self._per_chunk_features(dZ, self.q_dz)
        feats_ddz = self._per_chunk_features(ddZ, self.q_ddz)
        assert len(feats_z) == len(feats_dz) == len(feats_ddz)
        chunk_feats = [torch.cat([fz, fdz, fddz], dim=-1) for fz, fdz, fddz in zip(feats_z, feats_dz, feats_ddz)]  # each: [B, 12*d]
        if len(chunk_feats) == 0:
            # fallback to single chunk over whole sequence
            mean = Z.mean(dim=1); std = Z.std(dim=1, unbiased=False); maxv, _ = Z.max(dim=1)
            q_exp = self.q_z[None, None, :].expand(B, 1, d)
            scores = torch.einsum('btd,bsd->bts', Z, q_exp) / (d ** 0.5)
            w = torch.softmax(scores.squeeze(-1), dim=1)
            attn = torch.einsum('bt,btd->bd', w, Z)
            chunk_seq = torch.cat([mean, std, maxv, attn,
                                   mean, std, maxv, attn,
                                   mean, std, maxv, attn], dim=-1).unsqueeze(1)
        else:
            chunk_seq = torch.stack(chunk_feats, dim=1)  # [B, N_chunks, 12*d]

        if self.head_arch == "mlp":
            # Flatten across chunks
            flat = chunk_seq.reshape(B, -1)
            if self._mlp_head is None:
                in_dim = flat.size(-1)
                self._mlp_head = nn.Sequential(
                    nn.Linear(in_dim, self._cfg['hidden']), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(self._cfg['hidden'], self._cfg['out_dim'])
                ).to(flat.device)
            y = self._mlp_head(flat)
            return torch.clamp(y, 0.0, 1.0)

        if self.head_arch == "tcn":
            C_in = chunk_seq.size(-1)
            if self._tcn is None:
                layers = []
                C = self._cfg['tcn_channels']
                last_c = C_in
                for i in range(self._cfg['tcn_blocks']):
                    dilation = self._cfg['tcn_dilations'][min(i, len(self._cfg['tcn_dilations'])-1)]
                    layers += [
                        nn.Conv1d(last_c, C, kernel_size=3, padding=dilation, dilation=dilation),
                        nn.GELU(), nn.Dropout(self.dropout)
                    ]
                    last_c = C
                self._tcn = nn.Sequential(*layers).to(chunk_seq.device)
                self._tcn_q = nn.Parameter(torch.randn(C, device=chunk_seq.device))
                self._final_mlp = nn.Sequential(
                    nn.Linear(C, self._cfg['hidden']), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(self._cfg['hidden'], self._cfg['out_dim'])
                ).to(chunk_seq.device)
            x = chunk_seq.transpose(1, 2)  # [B, C_in, Tch]
            h = self._tcn(x)               # [B, C, Tch]
            q = self._tcn_q[None, :, None]
            scores = (h * q).sum(dim=1) / (h.size(1) ** 0.5)  # [B, Tch]
            w = torch.softmax(scores, dim=1)
            pooled = torch.einsum('bt,bct->bc', w, h)         # [B, C]
            y = self._final_mlp(pooled)
            return torch.clamp(y, 0.0, 1.0)

        if self.head_arch == "gru":
            F_in = chunk_seq.size(-1)
            if self._gru is None:
                self._gru = nn.GRU(input_size=F_in, hidden_size=self._cfg['gru_hidden'], num_layers=self._cfg['gru_layers'], batch_first=True, dropout=0.0).to(chunk_seq.device)
                self._gru_q = nn.Parameter(torch.randn(self._cfg['gru_hidden'], device=chunk_seq.device))
                self._final_mlp = nn.Sequential(
                    nn.Linear(self._cfg['gru_hidden'], self._cfg['hidden']), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(self._cfg['hidden'], self._cfg['out_dim'])
                ).to(chunk_seq.device)
            h, _ = self._gru(chunk_seq)  # [B, Tch, H]
            q = self._gru_q[None, None, :].expand(h.size(0), 1, -1)
            scores = torch.einsum('btd,bsd->bts', h, q) / (h.size(-1) ** 0.5)
            w = torch.softmax(scores.squeeze(-1), dim=1)
            pooled = torch.einsum('bt,btd->bd', w, h)
            y = self._final_mlp(pooled)
            return torch.clamp(y, 0.0, 1.0)

        if self.head_arch == "transformer":
            F_in = chunk_seq.size(-1)
            if self._tf_in is None:
                self._tf_in = nn.Linear(F_in, self._cfg['tf_d_model']).to(chunk_seq.device)
                encoder_layer = nn.TransformerEncoderLayer(d_model=self._cfg['tf_d_model'], nhead=self._cfg['tf_heads'], dim_feedforward=self._cfg['hidden'], dropout=self.dropout, batch_first=True)
                self._tf_enc = nn.TransformerEncoder(encoder_layer, num_layers=self._cfg['tf_layers']).to(chunk_seq.device)
                self._tf_q = nn.Parameter(torch.randn(self._cfg['tf_d_model'], device=chunk_seq.device))
                self._final_mlp = nn.Sequential(
                    nn.Linear(self._cfg['tf_d_model'], self._cfg['hidden']), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(self._cfg['hidden'], self._cfg['out_dim'])
                ).to(chunk_seq.device)
            z = self._tf_in(chunk_seq)
            h = self._tf_enc(z)  # [B, Tch, d_model]
            q = self._tf_q[None, None, :].expand(h.size(0), 1, -1)
            scores = torch.einsum('btd,bsd->bts', h, q) / (h.size(-1) ** 0.5)
            w = torch.softmax(scores.squeeze(-1), dim=1)
            pooled = torch.einsum('bt,btd->bd', w, h)
            y = self._final_mlp(pooled)
            return torch.clamp(y, 0.0, 1.0)

        # default to mlp
        flat = chunk_seq.reshape(B, -1)
        if self._mlp_head is None:
            in_dim = flat.size(-1)
            self._mlp_head = nn.Sequential(
                nn.Linear(in_dim, self._cfg['hidden']), nn.GELU(), nn.Dropout(self.dropout),
                nn.Linear(self._cfg['hidden'], self._cfg['out_dim'])
            ).to(flat.device)
        y = self._mlp_head(flat)
        return torch.clamp(y, 0.0, 1.0)


@torch.no_grad()
def _evaluate_richpool(model, loader, device="cpu"):
    ys, ps = [], []
    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    if not ys:
        return {}
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return _metrics(y, p)


def _fit_standardizer(Z_tr: np.ndarray):
    # Z_tr: [N,T,d]
    mu = Z_tr.mean(axis=(0, 1), keepdims=True)
    sd = Z_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def _apply_standardizer(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((Z - mu) / sd).astype(np.float32)


def _pearson_avg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred, target: [B, D]
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    num = (pred * target).sum(dim=0)
    den = torch.sqrt((pred.pow(2).sum(dim=0) + 1e-8) * (target.pow(2).sum(dim=0) + 1e-8))
    r = num / den
    return r.mean()


def _fit_affine_calibration(pred: np.ndarray, targ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-dimension affine calibration y ≈ a * p + b via least squares.
    pred, targ: [N, D]
    Returns (a: [D], b: [D])
    """
    assert pred.shape == targ.shape and pred.ndim == 2
    p_mean = pred.mean(axis=0)
    t_mean = targ.mean(axis=0)
    p0 = pred - p_mean
    t0 = targ - t_mean
    var_p = (p0 ** 2).mean(axis=0) + 1e-12
    cov_pt = (p0 * t0).mean(axis=0)
    a = cov_pt / var_p
    b = t_mean - a * p_mean
    return a.astype(np.float32), b.astype(np.float32)


def _apply_affine_calibration(pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (pred * a[None, :]) + b[None, :]


@torch.no_grad()
def _val_stats(model, loader, loss_type: str, alpha: float, device: str = "cpu"):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n_seen = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        # loss
        if loss_type == "ccc":
            pred_c = pred - pred.mean(dim=0, keepdim=True)
            y_c = yb - yb.mean(dim=0, keepdim=True)
            cov = (pred_c * y_c).mean(dim=0)
            var_p = pred.var(dim=0, unbiased=False) + 1e-8
            var_y = yb.var(dim=0, unbiased=False) + 1e-8
            mu_p = pred.mean(dim=0)
            mu_y = yb.mean(dim=0)
            ccc = 2 * cov / (var_p + var_y + (mu_p - mu_y).pow(2))
            loss = 1 - ccc.mean()
        else:
            mse = torch.nn.functional.mse_loss(pred, yb)
            r = _pearson_avg(pred, yb)
            loss = mse - alpha * r
        total_loss += float(loss.item()) * xb.size(0)
        n_seen += xb.size(0)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    if n_seen == 0:
        return float("inf"), {}
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    metrics = _metrics(y, p)
    return total_loss / max(1, n_seen), metrics


def train_richpool_mlp(
    X_train_seq: np.ndarray,
    Y_train: np.ndarray,
    X_test_seq: np.ndarray,
    Y_test: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.3,
    loss_type: str = "mse_r",  # "mse_r" or "ccc"
    alpha: float = 0.1,
    chunk_splits: tuple[int, ...] = (1, 2, 4),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    d_model = X_train_seq.shape[2]
    # Use test set as validation; fit standardization on full train
    mu, sd = _fit_standardizer(X_train_seq)
    X_tr_std = _apply_standardizer(X_train_seq, mu, sd)
    X_te_std = _apply_standardizer(X_test_seq, mu, sd)

    ds_tr = SequenceDataset(X_tr_std, Y_train)
    ds_va = SequenceDataset(X_te_std, Y_test)  # validation is test
    ds_te = SequenceDataset(X_te_std, Y_test)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ChunkedRichPoolMLP(d=d_model, out_dim=Y_train.shape[1], hidden=512, dropout=dropout, chunk_splits=chunk_splits).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_train = float('inf')
    epochs_since_improve = 0
    min_epochs = max(1, int(epochs))
    max_epochs = min_epochs + 200

    ep = 0
    while True:
        ep += 1
        model.train()
        loss_acc = 0.0
        n_seen = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            with torch.enable_grad():
                pred = model(xb)
                if loss_type == "ccc":
                    pred_c = pred - pred.mean(dim=0, keepdim=True)
                    y_c = yb - yb.mean(dim=0, keepdim=True)
                    cov = (pred_c * y_c).mean(dim=0)
                    var_p = pred.var(dim=0, unbiased=False) + 1e-8
                    var_y = yb.var(dim=0, unbiased=False) + 1e-8
                    mu_p = pred.mean(dim=0)
                    mu_y = yb.mean(dim=0)
                    ccc = 2 * cov / (var_p + var_y + (mu_p - mu_y).pow(2))
                    loss = 1 - ccc.mean()
                else:
                    mse = torch.nn.functional.mse_loss(pred, yb)
                    r = _pearson_avg(pred, yb)
                    loss = mse - alpha * r
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            loss_acc += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        sched.step()
        train_loss = loss_acc / max(1, n_seen)

        # Validation statistics
        val_loss, val_metrics = _val_stats(model, dl_va, loss_type, alpha, device=device)
        val_r = val_metrics.get("pearson_avg", float('nan'))
        print(f"[RichPool] epoch {ep:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val r {val_r:.3f}")

        # Early stopping on training loss with patience=5 after min_epochs
        if train_loss < best_train - 1e-6:
            best_train = train_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if ep >= min_epochs and epochs_since_improve >= 5:
            break
        if ep >= max_epochs:
            break

    te_metrics = _evaluate_richpool(model, dl_te, device=device)
    print(f"[RichPool] TEST: MSE_avg={te_metrics.get('mse_avg', float('nan')):.4f}, MAE_avg={te_metrics.get('mae_avg', float('nan')):.4f}, R2_avg={te_metrics.get('r2_avg', float('nan')):.3f}, r_avg={te_metrics.get('pearson_avg', float('nan')):.3f}")
    return model, te_metrics


# ========================= Finetuning (end-to-end) =========================
class FinetuneSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq: np.ndarray, Y: np.ndarray):
        assert X_seq.ndim == 3
        assert len(X_seq) == len(Y)
        self.X = X_seq.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _fit_input_standardizer(X_tr_seq: np.ndarray):
    # X_tr_seq: [N,T,D]
    mu = X_tr_seq.mean(axis=(0, 1), keepdims=True)
    sd = X_tr_seq.std(axis=(0, 1), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def _apply_input_standardizer(X_seq: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X_seq - mu) / sd).astype(np.float32)


class FinetuneRegressor(nn.Module):
    def __init__(self, cfg, out_dim: int = 3, hidden: int = 512, dropout: float = 0.3,
                 chunk_splits: tuple[int, ...] = (1, 2, 4), tokens_per_second: int | None = None,
                 head_arch: str = "mlp",
                 tcn_channels: int = 128, tcn_blocks: int = 2, tcn_dilations: tuple[int, ...] = (1, 2, 4),
                 gru_hidden: int = 128, gru_layers: int = 1,
                 tf_d_model: int = 128, tf_heads: int = 4, tf_layers: int = 1):
        super().__init__()
        # Keep cfg for reconstruction when saving/loading
        self.cfg = cfg
        self.backbone = PatchSeq2Seq(cfg)
        d = int(cfg["model"]["d_model"])
        self.head = ChunkedRichPoolMLP(
            d=d,
            out_dim=out_dim,
            hidden=hidden,
            dropout=dropout,
            chunk_splits=chunk_splits,
            tokens_per_second=tokens_per_second,
            head_arch=head_arch,
            tcn_channels=tcn_channels,
            tcn_blocks=tcn_blocks,
            tcn_dilations=tcn_dilations,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            tf_d_model=tf_d_model,
            tf_heads=tf_heads,
            tf_layers=tf_layers,
        )

    def forward(self, past_frames: torch.Tensor) -> torch.Tensor:
        enc_tokens = self.backbone.patch_embed_enc(past_frames)
        enc_out = self.backbone.encoder(enc_tokens)
        return self.head(enc_out)


@torch.no_grad()
def _evaluate_regression(model: nn.Module, loader, device="cpu"):
    ys, ps = [], []
    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    if not ys:
        return {}
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return _metrics(y, p)


def finetune_model(
    ckpt_loc: str,
    X_train_seq: np.ndarray,
    Y_train: np.ndarray,
    X_test_seq: np.ndarray,
    Y_test: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 5e-2,
    loss_type: str = "ccc",
    alpha: float = 0.1,
    head_first_epochs: int = 10,
    head_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    head_dropout: float = 0.5,
    chunk_splits: tuple[float, ...] = (1, 2, 4),
    frac_encoders_unfrozen: float = 1.0,
    calibrate: bool = False,
    head_arch: str = "mlp",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load checkpoint + cfg
    state = torch.load(ckpt_loc, map_location="cpu")
    cfg = state["config"]
    # Determine tokens_per_second from cfg: encoder tokens correspond to past_len/patch_size tokens over past_len seconds
    Tp = int(cfg["data"]["past_len"])  # frames
    P  = int(cfg["model"]["patch_size"])  # frames per token
    fps = 30  # dataset sampling assumed 30Hz unless specified; tokens/sec = (Tp/P)/(Tp/fps) = fps/P
    tokens_per_second = max(1, fps // max(1, P))

    model = FinetuneRegressor(
        cfg, out_dim=Y_train.shape[1], hidden=512, dropout=head_dropout,
        chunk_splits=tuple(float(s) for s in chunk_splits), tokens_per_second=tokens_per_second,
        head_arch=head_arch
    ).to(device)
    sd = state["model"]
    try:
        model.backbone.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.backbone.load_state_dict(sd, strict=True)

    # Standardize inputs on train only
    mu_in, sd_in = _fit_input_standardizer(X_train_seq)
    X_tr = _apply_input_standardizer(X_train_seq, mu_in, sd_in)
    X_te = _apply_input_standardizer(X_test_seq, mu_in, sd_in)

    # Attach standardizer and tokens_per_second to model for saving
    try:
        import numpy as _np
        model.input_standardizer_mu = torch.from_numpy(mu_in)
        model.input_standardizer_sd = torch.from_numpy(sd_in)
    except Exception:
        # Fallback: store as CPU tensors if conversion fails
        model.input_standardizer_mu = torch.tensor(mu_in)
        model.input_standardizer_sd = torch.tensor(sd_in)
    model.tokens_per_second = int(tokens_per_second)

    ds_tr = FinetuneSequenceDataset(X_tr, Y_train)
    ds_va = FinetuneSequenceDataset(X_te, Y_test)  # use test as val per current setup
    ds_te = FinetuneSequenceDataset(X_te, Y_test)

    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Stage 1: train head only ---
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt_head = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=max(1, head_first_epochs))
    for ep in range(1, min(head_first_epochs, epochs) + 1):
        model.train()
        loss_acc = 0.0
        n_seen = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            with torch.enable_grad():
                pred = model(xb)
                if loss_type == "ccc":
                    pred_c = pred - pred.mean(dim=0, keepdim=True)
                    y_c = yb - yb.mean(dim=0, keepdim=True)
                    cov = (pred_c * y_c).mean(dim=0)
                    var_p = pred.var(dim=0, unbiased=False) + 1e-8
                    var_y = yb.var(dim=0, unbiased=False) + 1e-8
                    mu_p = pred.mean(dim=0)
                    mu_y = yb.mean(dim=0)
                    ccc = 2 * cov / (var_p + var_y + (mu_p - mu_y).pow(2))
                    loss = 1 - ccc.mean()
                else:
                    mse = torch.nn.functional.mse_loss(pred, yb)
                    r = _pearson_avg(pred, yb)
                    loss = mse - alpha * r
                opt_head.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                opt_head.step()
            loss_acc += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        sched_head.step()
        train_loss = loss_acc / max(1, n_seen)
        val_loss, val_metrics = _val_stats(model, dl_va, loss_type, alpha, device=device)
        val_r = val_metrics.get("pearson_avg", float('nan'))
        print(f"[Finetune:H] epoch {ep:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val r {val_r:.3f}")

    # --- Stage 2: unfreeze encoder and train end-to-end ---
    # Freeze everything in backbone, then selectively unfreeze last K encoder blocks
    for p in model.backbone.parameters():
        p.requires_grad = False
    try:
        enc_layers = list(model.backbone.encoder.layers)
        n_layers = len(enc_layers)
        k = int(round(max(0.0, min(1.0, float(frac_encoders_unfrozen))) * n_layers))
        if k > 0:
            for blk in enc_layers[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # also unfreeze encoder final layernorm if present
            if hasattr(model.backbone.encoder, "final_ln"):
                for p in model.backbone.encoder.final_ln.parameters():
                    p.requires_grad = True
        print(f"[Finetune] Unfreezing {k}/{n_layers} encoder blocks (frac={frac_encoders_unfrozen})")
    except Exception:
        # Fallback: unfreeze all if something unexpected
        for p in model.backbone.parameters():
            p.requires_grad = True
        print("[Finetune] Warning: could not partially unfreeze encoder; unfreezing all backbone params.")

    # Head always trainable
    for p in model.head.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt_all = torch.optim.AdamW(params, lr=finetune_lr, weight_decay=weight_decay)
    rem_epochs = max(0, epochs - head_first_epochs)
    sched_all = torch.optim.lr_scheduler.CosineAnnealingLR(opt_all, T_max=max(1, rem_epochs))
    for ep2 in range(1, rem_epochs + 1):
        model.train()
        loss_acc = 0.0
        n_seen = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            with torch.enable_grad():
                pred = model(xb)
                if loss_type == "ccc":
                    pred_c = pred - pred.mean(dim=0, keepdim=True)
                    y_c = yb - yb.mean(dim=0, keepdim=True)
                    cov = (pred_c * y_c).mean(dim=0)
                    var_p = pred.var(dim=0, unbiased=False) + 1e-8
                    var_y = yb.var(dim=0, unbiased=False) + 1e-8
                    mu_p = pred.mean(dim=0)
                    mu_y = yb.mean(dim=0)
                    ccc = 2 * cov / (var_p + var_y + (mu_p - mu_y).pow(2))
                    loss = 1 - ccc.mean()
                else:
                    mse = torch.nn.functional.mse_loss(pred, yb)
                    r = _pearson_avg(pred, yb)
                    loss = mse - alpha * r
                opt_all.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_all.step()
            loss_acc += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        sched_all.step()
        train_loss = loss_acc / max(1, n_seen)
        val_loss, val_metrics = _val_stats(model, dl_va, loss_type, alpha, device=device)
        val_r = val_metrics.get("pearson_avg", float('nan'))
        print(f"[Finetune:FT] epoch {ep2:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val r {val_r:.3f}")

    # Optionally, final affine calibration on held-out 5% of test
    te_metrics = _evaluate_regression(model, dl_te, device=device)
    if calibrate:
        # build numpy preds/targets on test
        preds_list, targs_list = [], []
        for xb, yb in dl_te:
            xb = xb.to(device)
            with torch.no_grad():
                yhat = model(xb).cpu().numpy()
            preds_list.append(yhat)
            targs_list.append(yb.numpy())
        P = np.concatenate(preds_list, axis=0)
        T = np.concatenate(targs_list, axis=0)
        n = P.shape[0]
        cut = max(1, int(0.05 * n))
        # Last 5% for calibration to avoid leakage back into training loop
        P_tr, T_tr = P[:-cut], T[:-cut]
        P_ca, T_ca = P[-cut:], T[-cut:]
        a, b = _fit_affine_calibration(P_tr, T_tr)
        P_ca_adj = _apply_affine_calibration(P_ca, a, b)
        P_full_adj = _apply_affine_calibration(P, a, b)
        te_metrics = _metrics(T, P_full_adj)
        print(f"[Calibrate] Used {cut} samples for affine calibration; updated test metrics computed.")
    print(f"[Finetune] TEST: MSE_avg={te_metrics.get('mse_avg', float('nan')):.4f}, MAE_avg={te_metrics.get('mae_avg', float('nan')):.4f}, R2_avg={te_metrics.get('r2_avg', float('nan')):.3f}, r_avg={te_metrics.get('pearson_avg', float('nan')):.3f}")
    return model, te_metrics


def main():
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Train regressors on embeddings for eye gaze emotion dataset")
    parser.add_argument("--data_folder", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for MLP (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for MLP training (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for MLP (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for MLP (default: 1e-4)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for the results (default: None)")
    parser.add_argument("--output-name", type=str, default="glass", help="Output name for the results (default: glass)")
    parser.add_argument("--calibrate", default=False, help="Enable 5% affine calibration on test")
    parser.add_argument("--head-arch", type=str, default=None, help="Head architecture (default: tcn)")
    parser.add_argument("--chunk-split", type=float, default=None, help="Chunk split for the model (default: None)")
    parser.add_argument("--ckpt-loc", type=str, default=None, help="Checkpoint location for the model (default: None)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator (default: 42)")
    parser.add_argument("--model-size", type=str, default="small", help="Model size (default: large)")
    parser.add_argument("--save_model", action="store_true", default=False, help="Save trained model checkpoint to output-dir")
    args = parser.parse_args()
    
    # 0) Load dataset from CSV files
    train_csv = args.data_folder / "train.csv"
    test_csv = args.data_folder / "test.csv"
    X_train, Y_train, Z_train, X_test, Y_test, Z_test = load_csv_dataset(train_csv, test_csv, toggle_face=True)

    print(f"[data] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("[data] No samples found. Check your CSV files.")
    
    if args.ckpt_loc is None:
        ckpt_loc = "/data2/mjma/voices/self_supervised_gaze/checkpoints/large_5_sec_5_sec/best.ckpt"
    else:
        ckpt_loc = args.ckpt_loc
    
    train_eye_data = unroll_eye_data(X_train)
    test_eye_data = unroll_eye_data(X_test)

    # train_embeddings, _ = get_embeddings(ckpt_loc, train_eye_data)
    # test_embeddings, _ = get_embeddings(ckpt_loc, test_eye_data)

    # print(f"[data] Train embeddings: {train_embeddings.shape} | Test embeddings: {test_embeddings.shape}")

    # # Rich pooling + MLP on sequence embeddings with standardization
    # print("\n" + "="*50)
    # print("TRAINING (A) RichPoolMLP ON FROZEN SEQUENCE EMBEDDINGS")
    # print("="*50)
    # model, metrics = train_richpool_mlp(
    #     train_embeddings, Y_train, test_embeddings, Y_test,
    #     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=1e-2,
    #     dropout=0.3, loss_type="ccc", alpha=0.1
    # )

    # # Detailed per-dim metrics
    # print(f"[RichPool] per-dim MSE: {metrics['mse']} | MAE: {metrics['mae']} | R2: {metrics['r2']} | r: {metrics['pearson']}")

    # ---------------------------- Hyperparameters ----------------------------
    # Finetuning (end-to-end backbone + chunked head)
    BATCH_SIZE = int(args.batch_size)
    FT_WEIGHT_DECAY = 9e-2
    HEAD_FIRST_EPOCHS = 0
    EPOCHS = int(args.epochs)
    HEAD_LR = 1e-3
    FT_LR = 1e-4
    HEAD_DROPOUT = 0.15
    LOSS_TYPE = "mse_r"  # "ccc" or "mse_r"
    ALPHA = 0.1
    if args.chunk_split is None:
        CHUNK_SPLITS = (1,)
    else:
        CHUNK_SPLITS = (args.chunk_split,) # for legacy reasons, needs to be a tuple
    FRAC_ENCODERS_UNFROZEN =1  # 0.0 -> freeze all encoder blocks; 1.0 -> unfreeze all
    CALIBRATE = bool(args.calibrate)
    if args.head_arch is None:
        HEAD_ARCH = "tcn"  # one of {"mlp","tcn","gru","transformer"}
    else:
        HEAD_ARCH = args.head_arch
    # Optional head params (used when applicable)
    TCN_CHANNELS = 128
    TCN_BLOCKS = 2
    TCN_DILATIONS = (1, 2, 4)
    GRU_HIDDEN = 128
    GRU_LAYERS = 1
    TF_D_MODEL = 128
    TF_HEADS = 4
    TF_LAYERS = 1
    # ------------------------------------------------------------------------

    # (B) Finetune backbone + chunked head directly on raw sequences
    print("\n" + "="*50)
    print("TRAINING (B) FINETUNED BACKBONE + CHUNKED HEAD")
    print("="*50)
    # Reproducibility
    set_seed(args.seed)
    ft_model, ft_metrics = finetune_model(
        ckpt_loc,
        np.array(train_eye_data, dtype=np.float32), Y_train,
        np.array(test_eye_data, dtype=np.float32), Y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=FT_LR, weight_decay=FT_WEIGHT_DECAY,
        loss_type=LOSS_TYPE, alpha=ALPHA,
        head_first_epochs=HEAD_FIRST_EPOCHS, head_lr=HEAD_LR, finetune_lr=FT_LR, head_dropout=HEAD_DROPOUT,
        chunk_splits=CHUNK_SPLITS, frac_encoders_unfrozen=FRAC_ENCODERS_UNFROZEN, calibrate=CALIBRATE,
        head_arch=HEAD_ARCH
    )
    print(f"[Finetune] per-dim MSE: {ft_metrics['mse']} | MAE: {ft_metrics['mae']} | R2: {ft_metrics['r2']} | r: {ft_metrics['pearson']}")

    # ---------------------------- Save results JSON ----------------------------
    from datetime import datetime
    import json
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not args.output_dir:
        results_dir = Path("results") / args.data_folder
    else:
        results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    glass_record = {
        'model': 'GLASS_FINETUNE',
        'timestamp': timestamp,
        'mse_avg': float(ft_metrics.get('mse_avg', float('nan'))),
        'mae_avg': float(ft_metrics.get('mae_avg', float('nan'))),
        'r2_avg': float(ft_metrics.get('r2_avg', float('nan'))),
        'pearson_avg': float(ft_metrics.get('pearson_avg', float('nan'))),
        'mse_valence': float(ft_metrics['mse'][0]) if 'mse' in ft_metrics else None,
        'mse_arousal': float(ft_metrics['mse'][1]) if 'mse' in ft_metrics else None,
        'mse_dominance': float(ft_metrics['mse'][2]) if 'mse' in ft_metrics else None,
        'mae_valence': float(ft_metrics['mae'][0]) if 'mae' in ft_metrics else None,
        'mae_arousal': float(ft_metrics['mae'][1]) if 'mae' in ft_metrics else None,
        'mae_dominance': float(ft_metrics['mae'][2]) if 'mae' in ft_metrics else None,
        'r2_valence': float(ft_metrics['r2'][0]) if 'r2' in ft_metrics else None,
        'r2_arousal': float(ft_metrics['r2'][1]) if 'r2' in ft_metrics else None,
        'r2_dominance': float(ft_metrics['r2'][2]) if 'r2' in ft_metrics else None,
        'pearson_valence': float(ft_metrics['pearson'][0]) if 'pearson' in ft_metrics else None,
        'pearson_arousal': float(ft_metrics['pearson'][1]) if 'pearson' in ft_metrics else None,
        'pearson_dominance': float(ft_metrics['pearson'][2]) if 'pearson' in ft_metrics else None,
        # Hyperparameters snapshot
        'epochs': int(EPOCHS),
        'batch_size': int(BATCH_SIZE),
        'weight_decay': float(FT_WEIGHT_DECAY),
        'head_first_epochs': int(HEAD_FIRST_EPOCHS),
        'head_lr': float(HEAD_LR),
        'finetune_lr': float(FT_LR),
        'head_dropout': float(HEAD_DROPOUT),
        'loss_type': str(LOSS_TYPE),
        'alpha': float(ALPHA),
        'chunk_splits_seconds': list(map(float, CHUNK_SPLITS)),
        'frac_encoders_unfrozen': float(FRAC_ENCODERS_UNFROZEN),
    }

    combined = {
        'timestamp': timestamp,
        'data_folder': str(args.data_folder),
        'detailed_results': [glass_record],
        'summary': [{
            'model': glass_record['model'],
            'mse_avg': glass_record['mse_avg'],
            'mae_avg': glass_record['mae_avg'],
            'r2_avg': glass_record['r2_avg'],
            'pearson_avg': glass_record['pearson_avg'],
        }]
    }

    out_path = results_dir / f"{args.output_name}.json"
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\n[results] GLASS results saved to: {out_path}")

    # ---------------------------- Save model checkpoint ----------------------------
    if args.save_model:
        try:
            model_dir = results_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            # Ensure all lazy modules in the head are constructed by running a dummy forward
            try:
                with torch.no_grad():
                    Tp = int(ft_model.cfg["data"]["past_len"])
                    D_in = int(ft_model.cfg["model"]["D_in"])
                    device = next(ft_model.parameters()).device
                    _dummy = torch.zeros(1, Tp, D_in, dtype=torch.float32, device=device)
                    ft_model.eval()
                    _ = ft_model(_dummy)
            except Exception:
                pass
            # Build head kwargs snapshot for exact reconstruction
            head_cfg = getattr(ft_model.head, "_cfg", {})
            head_kwargs = {
                'out_dim': int(head_cfg.get('out_dim', 3)),
                'hidden': int(head_cfg.get('hidden', 512)),
                'dropout': float(getattr(ft_model.head, 'dropout', 0.3)),
                'chunk_splits_seconds': tuple(getattr(ft_model.head, 'chunk_seconds', (1,))),
                'tokens_per_second': int(getattr(ft_model, 'tokens_per_second', 1)),
                'head_arch': str(getattr(ft_model.head, 'head_arch', 'mlp')),
                'tcn_channels': int(head_cfg.get('tcn_channels', 128)),
                'tcn_blocks': int(head_cfg.get('tcn_blocks', 2)),
                'tcn_dilations': tuple(head_cfg.get('tcn_dilations', (1, 2, 4))),
                'gru_hidden': int(head_cfg.get('gru_hidden', 128)),
                'gru_layers': int(head_cfg.get('gru_layers', 1)),
                'tf_d_model': int(head_cfg.get('tf_d_model', 128)),
                'tf_heads': int(head_cfg.get('tf_heads', 4)),
                'tf_layers': int(head_cfg.get('tf_layers', 1)),
            }
            ckpt_payload = {
                'type': 'glass_finetune',
                'backbone_config': getattr(ft_model, 'cfg', None),
                'head_kwargs': head_kwargs,
                'state_dict': ft_model.state_dict(),
                'input_standardizer': {
                    'mu': getattr(ft_model, 'input_standardizer_mu', None).cpu().numpy() if hasattr(ft_model, 'input_standardizer_mu') else None,
                    'sd': getattr(ft_model, 'input_standardizer_sd', None).cpu().numpy() if hasattr(ft_model, 'input_standardizer_sd') else None,
                },
                'meta': {
                    'data_folder': str(args.data_folder),
                    'seed': int(args.seed),
                    'timestamp': timestamp,
                }
            }
            ckpt_path = model_dir / "model.ckpt"
            torch.save(ckpt_payload, ckpt_path)
            print(f"[model] Saved checkpoint to: {ckpt_path}")
        except Exception as e:
            print(f"[model] Failed to save checkpoint: {e}")
    
if __name__ == "__main__":
    main()