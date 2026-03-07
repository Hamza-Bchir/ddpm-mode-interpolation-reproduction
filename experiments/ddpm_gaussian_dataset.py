# experiments/plot_samples.py
import argparse
import os

import torch
import matplotlib.pyplot as plt

from src.models.mlp import NN
from src.ddpm import DDPM
from src.utils import GaussianDataset


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description=(
            "Generate samples from a trained DDPM checkpoint and automatically "
            "plot results depending on the dataset used during training. "
            "If the checkpoint was trained on gaussian1d, a histogram is created. "
            "If trained on gaussian2d, a 2D scatter plot is created."
        )
    )

    p.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/ddpm_gaussian.pt",
        help="Path to checkpoint file containing cfg and model_state_dict.",
    )

    p.add_argument(
        "--out",
        type=str,
        default="samples.png",
        help="Output image path.",
    )

    p.add_argument(
        "--n_gen",
        type=int,
        default=200_000,
        help="Number of generated samples for visualization.",
    )

    p.add_argument(
        "--bins",
        type=int,
        default=300,
        help="Number of histogram bins (used only for gaussian1d).",
    )

    p.add_argument(
        "--s",
        type=float,
        default=5.0,
        help="Marker size for scatter plot (used only for gaussian2d).",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency for histogram or scatter plot.",
    )

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    dataset_name = cfg.get("dataset", None)
    if dataset_name not in ["gaussian1d", "gaussian2d"]:
        raise ValueError(
            f"Unsupported dataset in checkpoint: {dataset_name}. "
            "Expected 'gaussian1d' or 'gaussian2d'."
        )

    dim = cfg["dim"]

    # Build model
    model = NN(
        in_features=cfg["in_features"],
        hidden_features=cfg["hidden_features"],
        t_features=cfg["t_features"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = DDPM(T=cfg["T"], device=device)

    # Real samples for comparison
    real_ds = GaussianDataset(n=args.n_gen, dim=dim, preset=True)
    real = real_ds.x.cpu()

    # Generated samples
    gen = diffusion.sample(model, n=args.n_gen, shape=dim, device=device).cpu()

    if dataset_name == "gaussian1d":
        real_1d = real.squeeze(1)
        gen_1d = gen.squeeze(1)

        plt.figure()
        plt.yscale("log")
        plt.hist(
            real_1d.numpy(),
            bins=args.bins,
            density=False,
            alpha=args.alpha,
            label="Real",
        )
        plt.hist(
            gen_1d.numpy(),
            bins=args.bins,
            density=False,
            alpha=args.alpha,
            label="Generated (DDPM)",
        )
        plt.grid(True)
        plt.legend()
        plt.title("Gaussian 1D: real vs generated")
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved histogram to: {args.out}")

    elif dataset_name == "gaussian2d":
        plt.figure()
        plt.scatter(
            real[:, 0].numpy(),
            real[:, 1].numpy(),
            s=args.s,
            alpha=args.alpha,
            label="Real",
        )
        plt.title("Gaussian 2D: real samples")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.legend()
        real_out = args.out.replace(".png", "_real.png")
        plt.savefig(real_out, dpi=200, bbox_inches="tight")
        print(f"Saved real scatter to: {real_out}")

        plt.figure()
        plt.scatter(
            gen[:, 0].numpy(),
            gen[:, 1].numpy(),
            s=args.s,
            alpha=args.alpha,
            label="Generated",
        )
        plt.title("Gaussian 2D: generated samples (DDPM)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.legend()
        gen_out = args.out.replace(".png", "_gen.png")
        plt.savefig(gen_out, dpi=200, bbox_inches="tight")
        print(f"Saved generated scatter to: {gen_out}")


if __name__ == "__main__":
    main()