import argparse
import json
import os
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.ddpm import DDPM
from src.models.mlp import NN
from src.utils import GaussianDataset, set_seed, hallucination_metric
from typing import Tuple


SEED = 0

@torch.no_grad()
def sample_with_x0_trajectory(
    diffusion,
    model,
    n,
    shape,
    record_last_k,
    device,
):
    """
    Generate samples while recording the predicted x0 trajectory
    for the last `record_last_k` reverse diffusion steps.

    Returns
    -------
    x_final : (n, *shape)
    x0_traj : (n, K, *shape)
    """

    if isinstance(shape, int):
        sample_shape = (n, shape)
    else:
        sample_shape = (n, *shape)

    x = torch.randn(sample_shape, device=device)

    traj = torch.empty((n, record_last_k, *sample_shape[1:]), device=device)

    for ti in reversed(range(diffusion.T)):

        t = torch.full((n,), ti, device=device, dtype=torch.long)

        x_prev, eps_pred = diffusion.p_sample(model, x, t)

        if ti < record_last_k:

            sqrt_ab = diffusion._extract(diffusion.sqrt_alphas_bar, t, x.shape)
            sqrt_1mab = diffusion._extract(diffusion.sqrt_one_minus_alphas_bar, t, x.shape)

            x0_hat = (x - sqrt_1mab * eps_pred) / (sqrt_ab + 1e-30)

            traj[:, ti] = x0_hat

        x = x_prev

    return x, traj



def train_ddpm(
    model,
    diffusion,
    train_loader,
    epochs,
    lr,
    device,
    print_every,
):

    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_time": [],
    }

    model.train()

    for ep in range(epochs):

        start = time.time()

        total_loss = 0.0
        n_batches = 0

        for it, (x0,) in enumerate(train_loader):

            x0 = x0.to(device)

            B = x0.size(0)

            t = torch.randint(0, diffusion.T, (B,), device=device)

            x_t, noise = diffusion.q_sample(x0, t)

            noise_pred = model(x_t, t)

            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (it + 1) % print_every == 0:
                tqdm.write(
                    f"[TRAIN] epoch {ep+1}/{epochs} "
                    f"iter {it+1}/{len(train_loader)} "
                    f"loss {loss.item():.6f}"
                )

        epoch_loss = total_loss / max(n_batches, 1)

        history["train_loss"].append(epoch_loss)
        history["train_time"].append(time.time() - start)

        tqdm.write(f"epoch {ep+1}/{epochs} | train {epoch_loss:.6f}")

    return history



def parse_args():

    p = argparse.ArgumentParser(
        description="Recursive training experiment on 2D Gaussian dataset."
    )

    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_generated", type=int, default=150_000)
    p.add_argument("--filtering", type=str, default="variance", choices=["variance", "random"])
    p.add_argument("--record_last_k", type=int, default=8)

    p.add_argument("--T", type=int, default=500)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_features", type=int, default=128)
    p.add_argument("--t_features", type=int, default=128)

    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--print_every", type=int, default=80)
    p.add_argument("--outdir", type=str, default="logs/rec_2d_variance")
    p.add_argument("--save_every_gen", action="store_true")

    return p.parse_args()


def main():

    args = parse_args()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.outdir, exist_ok=True)

    tqdm.write(f"device: {device}")
    tqdm.write(json.dumps(vars(args), indent=2))

    # True dataset

    true_dataset = GaussianDataset(n=args.n_train, dim=2, preset=True)

    current_train = true_dataset.x.clone()

    diffusion = DDPM(T=args.T, device=device)

    logs = {
        "args": vars(args),
        "seed": SEED,
        "generations": [],
    }

    # Recursive generations

    for g in range(args.generations):

        gen_dir = os.path.join(args.outdir, f"gen_{g:02d}")

        os.makedirs(gen_dir, exist_ok=True)

        tqdm.write("=" * 80)
        tqdm.write(f"Generation {g}")

        # Build model

        model = NN(
            in_features=2,
            hidden_features=args.hidden_features,
            t_features=args.t_features,
        ).to(device)

        train_loader = DataLoader(
            TensorDataset(current_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Train

        train_hist = train_ddpm(
            model=model,
            diffusion=diffusion,
            train_loader=train_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            print_every=args.print_every,
        )

        ckpt_path = os.path.join(gen_dir, "ddpm.pt")

        if args.save_every_gen or g == args.generations - 1:

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "generation": g,
                    "seed": SEED,
                },
                ckpt_path,
            )

            tqdm.write(f"Saved checkpoint: {ckpt_path}")

        if g == args.generations - 1:
            break

        # Sampling

        model.eval()

        x_gen, x_traj = sample_with_x0_trajectory(
            diffusion,
            model,
            args.n_generated,
            shape=2,
            record_last_k=args.record_last_k,
            device=device,
        )

        scores = hallucination_metric(
            x_traj,
            time_dim=1,
            reduce="mean",
        )

        # Filtering

        if args.filtering == "random":

            perm = torch.randperm(args.n_generated, device=device)

            keep_idx = perm[: args.n_train]

        else:

            keep_idx = torch.argsort(scores)[: args.n_train]

        next_train = x_gen[keep_idx].detach().cpu()

        # Save generation artifacts

        torch.save(
            {
                "generated": x_gen.cpu(),
                "trajectory": x_traj.cpu(),
                "scores": scores.cpu(),
                "selected": next_train,
            },
            os.path.join(gen_dir, "generated_data.pt"),
        )

        # Logging

        log_entry = {
            "generation": g,
            "train_loss": train_hist["train_loss"][-1],
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }

        logs["generations"].append(log_entry)

        with open(os.path.join(gen_dir, "log.json"), "w") as f:
            json.dump(log_entry, f, indent=2)

        current_train = next_train

    with open(os.path.join(args.outdir, "all_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    tqdm.write("Finished recursive training.")


if __name__ == "__main__":
    main()