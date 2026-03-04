# experiments/train_ddpm_handshapes_unet.py

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.ddpm import DDPM
from src.utils import set_seed, hallucination_metric, HandShapesDataset
from src.models.unet_ddpm_torch.unet import UNet


# -----------------------------
# I/O helpers
# -----------------------------
def make_run_dir(base_dir: str, run_name: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    out = os.path.join(base_dir, run_name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(out, "data_preview"), exist_ok=True)
    return out


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert images to [0,1] for visualization.
    Accepts x in [-1,1] or [0,1]. We clamp anyway.
    """
    # if it looks like [-1,1], map to [0,1]
    if x.min() < -0.1:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def save_image_grid(x: torch.Tensor, out_path: str, nrow: int = 8, title: str = "") -> None:
    """
    x: (B,1,H,W) in [0,1] preferred (but we clamp)
    """
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"Expected (B,1,H,W), got {tuple(x.shape)}")

    x = tensor_to_01(x.detach().cpu())
    B, _, H, W = x.shape
    nrow = int(nrow)
    ncol = int(np.ceil(B / nrow))

    canvas = torch.zeros((ncol * H, nrow * W), dtype=torch.float32)
    for i in range(B):
        r = i // nrow
        c = i % nrow
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = x[i, 0]

    plt.figure(figsize=(nrow, ncol))
    plt.imshow(canvas.numpy(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def default_record_timesteps(T: int, K: int) -> torch.Tensor:
    """
    Choose K timesteps with higher density at small t using log spacing.
    Returns unique timesteps in [0, T-1] as torch.long.
    """
    K = int(K)
    if K <= 0:
        raise ValueError("K must be > 0")

    ts = np.unique(np.logspace(np.log10(1), np.log10(T), K).astype(int) - 1)
    ts = np.clip(ts, 0, T - 1)
    return torch.tensor(ts, dtype=torch.long)


# -----------------------------
# Sampling + trajectory logging
# -----------------------------
@torch.no_grad()
def sample_with_x0_trajectory(
    diffusion: DDPM,
    model: nn.Module,
    n: int,
    shape: Tuple[int, int, int],
    device: torch.device,
    record_timesteps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full reverse chain, but record x_hat0 at a subset of timesteps.

    Returns:
      x_final:  (B,C,H,W)
      x0_traj:  (B,K,C,H,W) where K=len(recorded steps)
               ordered as encountered (high t -> low t)
    """
    model.eval()
    C, H, W = shape
    x = torch.randn((n, C, H, W), device=device)

    record_timesteps = record_timesteps.to(device=device, dtype=torch.long)
    record_set = set(int(t.item()) for t in record_timesteps)

    traj: List[torch.Tensor] = []

    for ti in reversed(range(diffusion.T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        x_prev, eps_pred = diffusion.p_sample(model, x, t)

        # x0_hat from epsilon prediction
        sqrt_ab = diffusion._extract(diffusion.sqrt_alphas_bar, t, x.shape)
        sqrt_1mab = diffusion._extract(diffusion.sqrt_one_minus_alphas_bar, t, x.shape)
        x0_hat = (x - sqrt_1mab * eps_pred) / torch.clamp(sqrt_ab, min=1e-8)

        if ti in record_set:
            traj.append(x0_hat.detach().clone())

        x = x_prev

    if len(traj) == 0:
        raise RuntimeError("Recorded trajectory is empty. Check record_timesteps vs diffusion.T.")

    x0_traj = torch.stack(traj, dim=1)  # (B,K,C,H,W)
    return x, x0_traj


# -----------------------------
# Train/eval
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    diffusion: DDPM,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    T: int,
    data_range: str,
    print_every: int,
) -> float:
    model.train()
    mse = nn.MSELoss()

    total = 0.0
    n_batches = 0

    for it, batch in enumerate(loader):
        # dataset may return (x,y) if return_label=True
        x0 = batch[0].to(device)

        if data_range == "minus_one_one":
            x0 = x0 * 2.0 - 1.0  # [0,1] -> [-1,1]

        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

        x_t, noise = diffusion.q_sample(x0, t)
        eps_pred = model(x_t, t)

        loss = mse(eps_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item())
        n_batches += 1

        if print_every > 0 and (it + 1) % print_every == 0:
            tqdm.write(f"[train] iter {it+1}/{len(loader)} loss={loss.item():.6f}")

    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: DDPM,
    loader: DataLoader,
    device: torch.device,
    T: int,
    data_range: str,
) -> float:
    model.eval()
    mse = nn.MSELoss()

    total = 0.0
    n_batches = 0

    for batch in loader:
        x0 = batch[0].to(device)

        if data_range == "minus_one_one":
            x0 = x0 * 2.0 - 1.0

        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

        x_t, noise = diffusion.q_sample(x0, t)
        eps_pred = model(x_t, t)
        loss = mse(eps_pred, noise)

        total += float(loss.item())
        n_batches += 1

    return total / max(n_batches, 1)


# -----------------------------
# Logging schema
# -----------------------------
@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    epoch_time_sec: float

    hall_mean: float
    hall_median: float
    hall_topk_mean: float
    hall_bottomk_mean: float

    path_samples_random: str
    path_samples_least_hall: str
    path_samples_most_hall: str


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM+UNet on discretized HandShapesDataset with hallucination logging.")

    # run/logging
    p.add_argument("--run_name", type=str, default=None, help="Run folder name under logs/shapes/. If None, uses timestamp.")
    p.add_argument("--log_root", type=str, default="logs/shapes", help="Base directory for logs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--print_every", type=int, default=0)
    p.add_argument("--save_ckpt_every", type=int, default=10)

    # diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule_type", type=str, default="linear")

    # data transform
    p.add_argument("--data_range", type=str, choices=["zero_one", "minus_one_one"], default="minus_one_one",
                   help="Whether to train on [0,1] images or mapped [-1,1].")

    # dataset size
    p.add_argument("--n_train", type=int, default=50_000)
    p.add_argument("--n_val", type=int, default=5_000)

    # sampling / hallucination logging
    p.add_argument("--sample_n", type=int, default=10, help="How many samples to generate each epoch.")
    p.add_argument("--hall_topk", type=int, default=8, help="How many most/least hallucinated samples to save.")
    p.add_argument("--traj_K", type=int, default=16, help="How many reverse steps to record for hallucination metric.")

    # UNet config (good defaults for 64x64 grayscale)
    p.add_argument("--hid_channels", type=int, default=64)
    p.add_argument("--ch_mult", type=str, default="1,2,4,8")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--apply_attn", type=str, default="0,0,1,1", help="Comma list of 0/1 per level.")

    # hand dataset config (your discretized setup)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--antialias", type=int, default=4)
    p.add_argument("--n_orientation_bins", type=int, default=3)
    p.add_argument("--n_columns", type=int, default=3)
    p.add_argument("--allowed_columns", type=str, default="left,right")

    # geometry (your smaller hand)
    p.add_argument("--palm_r", type=int, default=7)
    p.add_argument("--finger_len", type=int, default=8)
    p.add_argument("--finger_width", type=int, default=2)
    p.add_argument("--thumb_width_scale", type=float, default=1.0)

    return p.parse_args()


def parse_int_list(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_bool01_list(s: str) -> Tuple[bool, ...]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    out = []
    for v in vals:
        if v not in {"0", "1"}:
            raise ValueError(f"apply_attn expects 0/1 list, got '{v}' in '{s}'")
        out.append(v == "1")
    return tuple(out)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # run name
    if args.run_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"handshapes_unet_{ts}"

    run_dir = make_run_dir(args.log_root, args.run_name)
    samples_dir = os.path.join(run_dir, "samples")
    ckpt_dir = os.path.join(run_dir, "ckpts")

    # config dump
    save_json(os.path.join(run_dir, "config.json"), vars(args))

    # dataset
    allowed_cols = tuple(x.strip() for x in args.allowed_columns.split(",") if x.strip())
    ds_train = HandShapesDataset(
        n=args.n_train,
        image_size=args.image_size,
        antialias=args.antialias,
        return_label=True,  # ok; we ignore y
        position_mode="columns",
        n_columns=args.n_columns,
        allowed_columns=allowed_cols,
        enforce_in_bounds=True,
        center_jitter=0,
        orientation_mode="binned",
        n_orientation_bins=args.n_orientation_bins,
        palm_radius_range=(args.palm_r, args.palm_r),
        finger_len_range=(args.finger_len, args.finger_len),
        finger_width=args.finger_width,
        thumb_width_scale=args.thumb_width_scale,
        seed=args.seed,
    )
    ds_val = HandShapesDataset(
        n=args.n_val,
        image_size=args.image_size,
        antialias=args.antialias,
        return_label=True,
        position_mode="columns",
        n_columns=args.n_columns,
        allowed_columns=allowed_cols,
        enforce_in_bounds=True,
        center_jitter=0,
        orientation_mode="binned",
        n_orientation_bins=args.n_orientation_bins,
        palm_radius_range=(args.palm_r, args.palm_r),
        finger_len_range=(args.finger_len, args.finger_len),
        finger_width=args.finger_width,
        thumb_width_scale=args.thumb_width_scale,
        seed=args.seed + 10_000,
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # preview training data (grid)
    preview_x = torch.stack([ds_train[i][0] for i in range(36)], dim=0)  # (36,1,H,W)
    save_image_grid(preview_x, os.path.join(run_dir, "data_preview", "train_preview.png"), nrow=6, title="Train data preview")

    # diffusion
    diffusion = DDPM(T=args.T, device=device, schedule_type=args.schedule_type)

    # UNet
    ch_mult = parse_int_list(args.ch_mult)
    apply_attn = parse_bool01_list(args.apply_attn)
    if len(apply_attn) != len(ch_mult):
        raise ValueError(f"apply_attn length {len(apply_attn)} must match ch_mult length {len(ch_mult)}")

    model = UNet(
        in_channels=1,
        hid_channels=args.hid_channels,
        out_channels=1,
        ch_multipliers=ch_mult,
        num_res_blocks=args.num_res_blocks,
        apply_attn=apply_attn,
        drop_rate=0.0,
        resample_with_conv=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # log files
    history_jsonl = os.path.join(run_dir, "history.jsonl")
    history_csv = os.path.join(run_dir, "history.csv")
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([field for field in EpochLog.__dataclass_fields__.keys()])

    record_timesteps = default_record_timesteps(args.T, args.traj_K)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            loader=train_loader,
            opt=opt,
            device=device,
            T=args.T,
            data_range=args.data_range,
            print_every=args.print_every,
        )
        val_loss = evaluate(
            model=model,
            diffusion=diffusion,
            loader=val_loader,
            device=device,
            T=args.T,
            data_range=args.data_range,
        )

        # ---- sampling + hallucination metric
        x_final, x0_traj = sample_with_x0_trajectory(
            diffusion=diffusion,
            model=model,
            n=args.sample_n,
            shape=(1, args.image_size, args.image_size),
            device=device,
            record_timesteps=record_timesteps,
        )

        # hallucination scores per sample (B,)
        hall = hallucination_metric(x0_traj).detach().cpu()
        hall_sorted, idx = torch.sort(hall, descending=True)

        k = min(args.hall_topk, args.sample_n)
        idx_most = idx[:k]
        idx_least = idx[-k:]

        # convert to view space
        x_final_vis = tensor_to_01(x_final)
        x_least = x_final_vis[idx_least].detach().cpu()
        x_most = x_final_vis[idx_most].detach().cpu()
        x_rand = x_final_vis[:k].detach().cpu()

        p_rand = os.path.join(samples_dir, f"epoch_{epoch:04d}_random.png")
        p_least = os.path.join(samples_dir, f"epoch_{epoch:04d}_least_hall.png")
        p_most = os.path.join(samples_dir, f"epoch_{epoch:04d}_most_hall.png")

        save_image_grid(x_rand, p_rand, nrow=int(np.sqrt(k)) if int(np.sqrt(k)) > 0 else 4, title=f"Epoch {epoch} random")
        save_image_grid(x_least, p_least, nrow=int(np.sqrt(k)) if int(np.sqrt(k)) > 0 else 4, title=f"Epoch {epoch} least hallucinated")
        save_image_grid(x_most, p_most, nrow=int(np.sqrt(k)) if int(np.sqrt(k)) > 0 else 4, title=f"Epoch {epoch} most hallucinated")

        # stats
        hall_mean = float(hall.mean().item())
        hall_median = float(hall.median().item())
        hall_topk_mean = float(hall_sorted[:k].mean().item())
        hall_bottomk_mean = float(hall_sorted[-k:].mean().item())

        epoch_time = time.time() - t0

        row = EpochLog(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            epoch_time_sec=float(epoch_time),
            hall_mean=hall_mean,
            hall_median=hall_median,
            hall_topk_mean=hall_topk_mean,
            hall_bottomk_mean=hall_bottomk_mean,
            path_samples_random=os.path.relpath(p_rand, run_dir),
            path_samples_least_hall=os.path.relpath(p_least, run_dir),
            path_samples_most_hall=os.path.relpath(p_most, run_dir),
        )

        # append logs
        with open(history_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row)) + "\n")

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([getattr(row, k) for k in EpochLog.__dataclass_fields__.keys()])

        # checkpoints
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)},
                os.path.join(ckpt_dir, "best.pt"),
            )

        if args.save_ckpt_every > 0 and epoch % args.save_ckpt_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)},
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"),
            )

        tqdm.write(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"hall(mean/med)={hall_mean:.4e}/{hall_median:.4e} "
            f"topk={hall_topk_mean:.4e} bottomk={hall_bottomk_mean:.4e} "
            f"time={epoch_time:.1f}s"
        )

    print(f"\n[OK] Run saved to: {run_dir}")
    print("Key outputs:")
    print(" - config.json")
    print(" - history.jsonl (one JSON per epoch)")
    print(" - history.csv")
    print(" - data_preview/train_preview.png")
    print(" - samples/epoch_*.png (random / least_hall / most_hall each epoch)")
    print(" - ckpts/best.pt (+ periodic epoch_*.pt if enabled)")


if __name__ == "__main__":
    main()