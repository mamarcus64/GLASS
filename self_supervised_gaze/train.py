# --- top of train.py ---
import os
import math
import time
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

import argparse  # parse args before importing torch/cuda for gpu selection
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default=None, help="Path to YAML config")
ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids, e.g. '0,1' (overrides YAML)")
args = ap.parse_args([] if False else None)  # allow running in notebooks too

if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

# Now safe to import torch/cuda things and continue
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from models.patch_transformer import PatchSeq2Seq
from gaze_forecast_dataset import GazeForecastDataset

# ---------------- Utils ----------------

def safe_collate(batch):
    """
    batch: list of tuples (past, future, mp, mf, meta)
    Ensures meta always has 'skipped', and stacks tensors safely.
    """
    pasts   = []
    futures = []
    mps     = []
    mfs     = []

    videos = []
    starts = []
    skipped_flags = []

    for past, future, mp, mf, meta in batch:
        # Tensors
        pasts.append(past)
        futures.append(future)
        mps.append(mp)
        mfs.append(mf)

        # Meta (normalize keys)
        m = dict(meta) if isinstance(meta, dict) else {}
        videos.append(m.get("video", ""))
        starts.append(int(m.get("start_idx", 0)))
        skipped_flags.append(bool(m.get("skipped", False)))

    pasts   = torch.stack(pasts, dim=0)
    futures = torch.stack(futures, dim=0)
    mps     = torch.stack(mps, dim=0)
    mfs     = torch.stack(mfs, dim=0)

    meta_out = {
        "video": videos,                              # list[str]
        "start_idx": torch.tensor(starts, dtype=torch.long),
        "skipped": torch.tensor(skipped_flags, dtype=torch.bool),
    }
    return pasts, futures, mps, mfs, meta_out


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def velocity(x: torch.Tensor) -> torch.Tensor:
    # x: (B,T,D) -> (B,T-1,D)
    return x[:, 1:, :] - x[:, :-1, :]

def masked_huber(pred, target, mask, delta=1.0):
    valid = mask.unsqueeze(-1).expand_as(pred)
    diff = pred[valid] - target[valid]
    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    loss = 0.5 * quad**2 + delta * lin
    return loss.mean() if loss.numel() else torch.tensor(0.0, device=pred.device)

def masked_smoothl1(pred, target, mask, beta=1.0):
    valid = mask.unsqueeze(-1).expand_as(pred)
    return nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=beta) if valid.any() else torch.tensor(0.0, device=pred.device)

def temporal_tv_loss(x: torch.Tensor, mask: torch.Tensor):
    if x.size(1) < 2:
        return torch.tensor(0.0, device=x.device)
    m = mask[:, 1:] & mask[:, :-1]
    diff = (x[:, 1:, :] - x[:, :-1, :])[m]
    return diff.abs().mean() if diff.numel() else torch.tensor(0.0, device=x.device)

@torch.no_grad()
def masked_pearson_r(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Mean Pearson r across channels, using only frames where mask==True.
    pred/target: (B,T,D), mask: (B,T) bool
    """
    device = pred.device
    valid = mask.unsqueeze(-1).expand_as(pred)  # (B,T,D)
    if not valid.any():
        return 0.0
    p = pred[valid].view(-1)  # flatten over valid elements but we need per-D, so regroup
    t = target[valid].view(-1)

    # compute per-channel r: do it by reshaping:
    B, T, D = pred.shape
    pred_v = pred.masked_select(valid).view(-1, D)   # (N_valid_frames, D)
    targ_v = target.masked_select(valid).view(-1, D) # (N_valid_frames, D)

    # subtract mean per channel
    p_mean = pred_v.mean(dim=0, keepdim=True)
    t_mean = targ_v.mean(dim=0, keepdim=True)
    p0 = pred_v - p_mean
    t0 = targ_v - t_mean
    num = (p0 * t0).sum(dim=0)
    den = (p0.pow(2).sum(dim=0).sqrt() * t0.pow(2).sum(dim=0).sqrt()) + 1e-9
    r = (num / den).clamp(min=-1.0, max=1.0)  # (D,)
    return float(r.mean().item())

def save_checkpoint(state: Dict, ckpt_dir: Path, name: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    torch.save(state, path)
    return path

# --------------- Main -------------------

def main(cfg_path=None):
    if cfg_path is not None: # prioritize cfg_path over args
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    elif args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError("No config path provided (either directly in main or via CLI with --config)")

    set_seed(cfg["optim"]["seed"])

    # Dirs
    log_dir = Path(cfg["logging"]["log_dir"]) / cfg["logging"]["run_name"]
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"]) / cfg["logging"]["run_name"]
    
    if cfg["data"]["use_norm"]:
        log_dir = Path(str(log_dir) + '_normalized')
        ckpt_dir = Path(str(ckpt_dir) + '_normalized')
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))
    csv_log = open(log_dir / "metrics.csv", "w")
    csv_log.write("step,split,loss,coord_mae,coord_mse,vel_mae,val_accuracy_r\n"); csv_log.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (single dataset, we will split once)
    ds = GazeForecastDataset(
        raw_dir=cfg["data"]["raw_dir"],
        norm_dir=cfg["data"]["norm_dir"],
        use_norm=cfg["data"]["use_norm"],
        past_len=cfg["data"]["past_len"],
        future_len=cfg["data"]["future_len"],
        stride=cfg["data"]["stride"],
        min_valid_ratio=cfg["data"]["min_valid_ratio"],
        zero_eps=cfg["data"]["zero_eps"],
        repetitive_num_sampled=cfg["data"]["repetitive_num_sampled"],
        repetitive_fraction=cfg["data"]["repetitive_fraction"],
        channels=cfg["data"]["channels"],
        cache_arrays=cfg["data"].get("cache_arrays", False),
        mmap_mode=cfg["data"].get("mmap_mode", "r"),
        max_videos=cfg["data"].get("max_videos", None),
        shuffle_videos=cfg["data"].get("shuffle_videos", False),
        shuffle_seed=cfg["data"].get("shuffle_seed", 42),
        max_resample_attempts=cfg["data"].get("max_resample_attempts", 50),
    )

    # Split
    def subject_from_vid(vid: str) -> str:
        # expects "<subject>.<video_index>", e.g., "1234.2"
        # robust fallback: take everything before the first dot
        return vid.split(".", 1)[0]

    rng = np.random.default_rng(cfg["data"]["split_seed"])

    # Unique subjects from dataset videos
    subjects = sorted({subject_from_vid(v) for v in ds.videos})

    # Random subject split by ratio
    subs = subjects.copy()
    rng.shuffle(subs)
    n_val_subj = max(1, int(round(len(subs) * cfg["data"]["val_split"])))
    val_subjects = sorted(subs[:n_val_subj])

    train_subjects = sorted([s for s in subjects if s not in set(val_subjects)])

    # Map video -> subject
    vid2subj = {v: subject_from_vid(v) for v in ds.videos}

    # Collect indices for each split (by video’s subject)
    train_indices, val_indices = [], []
    for i, (vid, start) in enumerate(ds._index):
        subj = vid2subj[vid]
        if subj in val_subjects:
            val_indices.append(i)
        else:
            train_indices.append(i)

    from torch.utils.data import Subset
    train_ds = Subset(ds, train_indices)
    val_ds   = Subset(ds, val_indices)

    print(f"[Split] Subjects: total={len(subjects)} | train={len(train_subjects)} | val={len(val_subjects)}")
    print(f"        Train subjects (first 10): {train_subjects[:10]}")
    print(f"        Val subjects   (first 10): {val_subjects[:10]}")
    print(f"[Split] Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,}")

    # (nice to persist the split for reproducibility/debug)
    split_dir = log_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "train_subjects.txt", "w") as f: f.write("\n".join(train_subjects) + "\n")
    with open(split_dir / "val_subjects.txt", "w") as f:   f.write("\n".join(val_subjects) + "\n")
    with open(split_dir / "train_videos.txt", "w") as f:   f.write("\n".join(sorted({vid for vid in ds.videos if vid2subj[vid] in set(train_subjects)})) + "\n")
    with open(split_dir / "val_videos.txt", "w") as f:     f.write("\n".join(sorted({vid for vid in ds.videos if vid2subj[vid] in set(val_subjects)})) + "\n")
    
    

    num_workers = cfg["logging"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["optim"]["batch_size"], shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)

    print(f"[Data] Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,}")
    steps_in_full_pass = math.ceil(len(train_loader))  # single pass through all train windows
    print(f"[Train] Batch size: {cfg['optim']['batch_size']} | Steps in full pass: {steps_in_full_pass:,}")

    # Model
    model = PatchSeq2Seq(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] params: total={n_params:,} trainable={n_trainable:,}")
    for name, p in list(model.named_parameters())[:10]:
        print("  ", name, tuple(p.shape), "requires_grad=", p.requires_grad)
    assert n_trainable > 0, "Model has no trainable parameters (check module registration)."
    
    if torch.cuda.device_count() > 1 and cfg["logging"].get("data_parallel", True):
        model = nn.DataParallel(model)

    # Optimizer & Scheduler (cosine over planned steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                            betas=tuple(cfg["optim"]["betas"]),
                            weight_decay=cfg["optim"]["weight_decay"])

    # Determine total steps (for LR schedule)
    max_steps_cfg = cfg["optim"].get("max_steps", -1)
    total_steps = steps_in_full_pass if (max_steps_cfg is None or max_steps_cfg < 0) else min(max_steps_cfg, steps_in_full_pass)
    warmup_steps = cfg["optim"]["warmup_steps"]
    
    ss_cfg = cfg["optim"].get("sched_sampling", {})
    ss_enabled = bool(ss_cfg.get("enabled", False))
    ss_start = int(ss_cfg.get("start_step", 0))
    ss_end = ss_cfg.get("end_step", None)
    if ss_end is None:
        end_fraction = float(ss_cfg.get("end_fraction", 0.6))
        ss_end = max(ss_start + 1, int(end_fraction * total_steps))
    ss_min = float(ss_cfg.get("min_tf_prob", 0.0))

    def current_tf_prob(step):
        if not ss_enabled:
            return None  # use pure teacher-forced path
        if step <= ss_start:
            return 1.0
        if step >= ss_end:
            return ss_min
        # linear decay
        span = max(1, ss_end - ss_start)
        prob = 1.0 - (step - ss_start) / span
        return max(ss_min, min(1.0, prob))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # Loss config
    use_amp = cfg["optim"]["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_coord_type = cfg["loss"]["coord"]
    vel_w = float(cfg["loss"]["velocity_weight"])
    tv_w = float(cfg["loss"]["tv_weight"])

    validate_every = int(cfg["logging"].get("validate_every_steps", 1000))
    best_val_loss = float("inf")
    global_step = 0

    # -------- Validation helper --------
    @torch.no_grad()
    def run_validation(step_tag: int):
        nonlocal best_val_loss
        model.eval()

        agg_loss = agg_mae = agg_mse = agg_vmae = 0.0
        denom = 0

        all_pred, all_targ, all_mask = [], [], []
        all_pred_baseline = []  # last-frame baseline

        for batch in tqdm(val_loader, desc="validating", ncols=100, leave=False):
            past, future, mp, mf, meta = batch
            past = past.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            mf = mf.to(device, non_blocking=True)
            
            # Teacher-forcing is 0, want to do entirely autoregressive for validation
            pred = model(past, future, tf_prob=0)

            # ----- losses/metrics for model -----
            if loss_coord_type == "huber":
                coord_loss = masked_huber(pred, future, mf, delta=1.0)
            else:
                coord_loss = masked_smoothl1(pred, future, mf, beta=1.0)
            pred_v = velocity(pred)
            fut_v = velocity(future)
            mf_v = mf[:, 1:] & mf[:, :-1]
            vel_loss = masked_huber(pred_v, fut_v, mf_v, delta=1.0)
            tv_loss = temporal_tv_loss(pred, mf)
            loss = coord_loss + vel_w * vel_loss + tv_w * tv_loss

            with torch.no_grad():
                coord_mae = (pred[mf.unsqueeze(-1).expand_as(pred)] - future[mf.unsqueeze(-1).expand_as(future)]).abs().mean().item() if mf.any() else 0.0
                coord_mse = (pred[mf.unsqueeze(-1).expand_as(pred)] - future[mf.unsqueeze(-1).expand_as(future)]).pow(2).mean().item() if mf.any() else 0.0
                vel_mae   = (pred_v[mf_v.unsqueeze(-1).expand_as(pred_v)] - fut_v[mf_v.unsqueeze(-1).expand_as(fut_v)]).abs().mean().item() if mf_v.any() else 0.0

            agg_loss += float(loss.item()); agg_mae += coord_mae; agg_mse += coord_mse; agg_vmae += vel_mae
            denom += 1

            all_pred.append(pred.detach().cpu())
            all_targ.append(future.detach().cpu())
            all_mask.append(mf.detach().cpu())

            # ----- naive baseline: repeat last past frame -----
            baseline = past[:, -1:, :].expand_as(future)
            all_pred_baseline.append(baseline.detach().cpu())

        # ---- aggregate over val set ----
        val_loss = agg_loss / max(1, denom)
        val_mae  = agg_mae  / max(1, denom)
        val_mse  = agg_mse  / max(1, denom)
        val_vmae = agg_vmae / max(1, denom)

        pred_cat = torch.cat(all_pred, dim=0)
        targ_cat = torch.cat(all_targ, dim=0)
        mask_cat = torch.cat(all_mask, dim=0)
        base_cat = torch.cat(all_pred_baseline, dim=0)

        val_r   = masked_pearson_r(pred_cat, targ_cat, mask_cat)
        base_r  = masked_pearson_r(base_cat, targ_cat, mask_cat)

        # baseline MAE
        if mask_cat.any():
            base_mae = (base_cat[mask_cat.unsqueeze(-1).expand_as(base_cat)]
                        - targ_cat[mask_cat.unsqueeze(-1).expand_as(targ_cat)]).abs().mean().item()
        else:
            base_mae = 0.0

        # valid fraction and target per-channel std (on valid frames)
        valid_frac = float(mask_cat.float().mean().item())
        if mask_cat.any():
            tv = targ_cat[mask_cat.unsqueeze(-1).expand_as(targ_cat)].view(-1, targ_cat.size(-1))
            std_ch = tv.std(dim=0)
            writer.add_histogram("val/target_std_per_channel", std_ch, step_tag)

        # ---- log ----
        writer.add_scalar("val/loss", val_loss, step_tag)
        writer.add_scalar("val/accuracy", val_r, step_tag)
        writer.add_scalar("val/coord_mae", val_mae, step_tag)
        writer.add_scalar("val/coord_mse", val_mse, step_tag)
        writer.add_scalar("val/vel_mae", val_vmae, step_tag)
        writer.add_scalar("val/valid_frac", valid_frac, step_tag)

        writer.add_scalar("val/lf_baseline_r", base_r, step_tag)
        writer.add_scalar("val/lf_baseline_mae", base_mae, step_tag)

        csv_log.write(f"{step_tag},val,{val_loss:.6f},{val_mae:.6f},{val_mse:.6f},{val_vmae:.6f},{val_r:.6f}\n"); csv_log.flush()

        # ---- checkpoints ----
        state = {
            "step": step_tag,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
            "val_r": val_r,
            "baseline_r": base_r,
        }
        save_checkpoint(state, ckpt_dir, "last.ckpt")
        nonlocal best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(state, ckpt_dir, "best.ckpt")

        tag = "Autoregressive only"
        print(f"[VAL {tag} step {step_tag}] loss={val_loss:.4f} r={val_r:.4f} (baseline r={base_r:.4f}) "
            f"mae={val_mae:.4f} (baseline mae={base_mae:.4f}) valid={valid_frac:.2f}")
        model.train()
        return val_loss, val_r

    # -------- Single-pass training (with periodic val) --------
    model.train()
    coord_label = "huber" if loss_coord_type == "huber" else "smoothl1"
    print(f"[Train] Starting single-pass training | loss={coord_label} + {vel_w}*vel + {tv_w}*tv")
    pbar = tqdm(train_loader, desc="train (single pass)", ncols=100)
    for batch in pbar:
        # Stop if we hit max_steps cap (if set)
        if global_step >= total_steps:
            break

        past, future, mp, mf, meta = batch
        past = past.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        mf = mf.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            tf_prob = current_tf_prob(global_step)
            pred = model(past, future, tf_prob=tf_prob)
            
            if tf_prob is not None:
                writer.add_scalar("train/tf_prob", tf_prob, global_step)
            
            if loss_coord_type == "huber":
                coord_loss = masked_huber(pred, future, mf, delta=1.0)
            else:
                coord_loss = masked_smoothl1(pred, future, mf, beta=1.0)
            pred_v = velocity(pred)
            fut_v = velocity(future)
            mf_v = mf[:, 1:] & mf[:, :-1]
            vel_loss = masked_huber(pred_v, fut_v, mf_v, delta=1.0)
            tv_loss = temporal_tv_loss(pred, mf)
            loss = coord_loss + vel_w * vel_loss + tv_w * tv_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        global_step += 1

        # Train metrics (log every step)
        with torch.no_grad():
            coord_mae = (pred[mf.unsqueeze(-1).expand_as(pred)] - future[mf.unsqueeze(-1).expand_as(pred)]).abs().mean().item() if mf.any() else 0.0
            coord_mse = (pred[mf.unsqueeze(-1).expand_as(pred)] - future[mf.unsqueeze(-1).expand_as(pred)]).pow(2).mean().item() if mf.any() else 0.0
            vel_mae   = (velocity(pred)[(mf[:,1:] & mf[:,:-1]).unsqueeze(-1).expand_as(velocity(pred))] \
                        - velocity(future)[(mf[:,1:] & mf[:,:-1]).unsqueeze(-1).expand_as(velocity(future))]).abs().mean().item() if (mf[:,1:] & mf[:,:-1]).any() else 0.0

        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/coord_mae", coord_mae, global_step)
        writer.add_scalar("train/coord_mse", coord_mse, global_step)
        writer.add_scalar("train/vel_mae", vel_mae, global_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        csv_log.write(f"{global_step},train,{loss.item():.6f},{coord_mae:.6f},{coord_mse:.6f},{vel_mae:.6f},\n"); csv_log.flush()

        pbar.set_postfix(loss=f"{loss.item():.4f}", rlr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Periodic validation
        if (global_step % validate_every) == 0:
            run_validation(global_step)

    pbar.close()

    # Final validation at the end of the single pass (if we didn’t just do one)
    if (global_step % validate_every) != 0:
        run_validation(global_step)

    writer.close()
    csv_log.close()
    print(f"Done. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints → {ckpt_dir}")
    print(f"TensorBoard → {log_dir}")

if __name__ == "__main__":
    main()
