import argparse
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from src.models.mlp import NN
from src.ddpm import DDPM
from src.utils import set_seed

SEED = 0


def mog_score_closed_form(X, diffusion, t, mus=(1.0, 2.0, 3.0), sigma=0.05, weights=None):
    """
    Closed-form marginal score: ∂/∂x log q_t(x) for a 1D MoG under DDPM forward noising.
    """
    # t must be an int here (NOT a (B,) tensor)
    if isinstance(t, torch.Tensor):
        t = int(t.item())

    device = X.device
    dtype = X.dtype
    X = X.view(-1, 1)  # [N,1]
    K = len(mus)

    if weights is None:
        weights = [1.0 / K] * K
    if len(weights) != K:
        raise ValueError(f"weights length {len(weights)} must match mus length {K}")

    abar = diffusion.alphas_bar[t].to(device=device, dtype=dtype)          # ᾱ_t
    c_t  = torch.sqrt(abar)                                                # √ᾱ_t
    v_t  = (1.0 - abar) + abar * (sigma ** 2)                               # variance (scalar)

    mu0 = torch.tensor(mus, device=device, dtype=dtype).view(1, K)         # [1,K]
    w   = torch.tensor(weights, device=device, dtype=dtype).view(1, K)     # [1,K]
    mu_t = c_t * mu0                                                       # [1,K]

    xK  = X.expand(-1, K)                                                  # [N,K]
    muK = mu_t.expand(X.shape[0], -1)                                      # [N,K]

    # Gaussian pdf values up to proportionality (normalizing constant cancels in ratio)
    expo = torch.exp(- (xK - muK) ** 2 / (2.0 * v_t + 1e-30))               # [N,K]

    den = (w * expo).sum(dim=1, keepdim=True)                               # [N,1]
    num = (w * expo) * (-(xK - muK) / (v_t + 1e-30))                        # [N,K]
    score = num.sum(dim=1, keepdim=True) / (den + 1e-30)                    # [N,1]
    return score.view_as(X)                                                # [N,1]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description="Reproduce Figure 4: closed-form MoG score vs learned score from DDPM epsilon-prediction."
    )

    p.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/ddpm_gaussian1d.pt",
        help="Path to checkpoint containing cfg and model_state_dict.",
    )

    p.add_argument(
        "--out",
        type=str,
        default="figure4_scores.png",
        help="Output image path for the saved figure.",
    )

    p.add_argument(
        "--mus",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Means of the Gaussian mixture components at t=0 (space-separated). Example: --mus 1 2 3",
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Shared standard deviation of each Gaussian component at t=0 (default: %(default)s).",
    )

    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional mixture weights (space-separated). If omitted, uses uniform weights.",
    )

    p.add_argument(
        "--xmin",
        type=float,
        default=0.5,
        help="Minimum x value of evaluation grid (default: %(default)s).",
    )

    p.add_argument(
        "--xmax",
        type=float,
        default=3.5,
        help="Maximum x value of evaluation grid (default: %(default)s).",
    )

    p.add_argument(
        "--xstep",
        type=float,
        default=1e-3,
        help="Step size for evaluation grid (default: %(default)s).",
    )

    p.add_argument(
        "--t_stride",
        type=int,
        default=80,
        help="Stride for selecting timesteps: 1, 1+stride, ... (default: %(default)s).",
    )

    p.add_argument(
        "--t_start",
        type=int,
        default=1,
        help="First timestep to plot (inclusive) (default: %(default)s).",
    )

    p.add_argument(
        "--t_end",
        type=int,
        default=None,
        help="Last timestep bound (exclusive). If None, uses cfg['T'].",
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="If set, also display the figure interactively.",
    )

    args = p.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    dataset_name = cfg.get("dataset", None)
    if dataset_name != "gaussian1d":
        raise ValueError(
            f"Unsupported dataset in checkpoint: {dataset_name}. Expected 'gaussian1d'."
        )

    # Build model
    model = NN(
        in_features=cfg["in_features"],
        hidden_features=cfg["hidden_features"],
        t_features=cfg["t_features"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = DDPM(T=cfg["T"], device=device)

    # Evaluation grid is x_t (not x0). For Fig 4, we want score as function of x at timestep t.
    X = torch.arange(args.xmin, args.xmax, args.xstep, device=device).view(-1, 1)  # [N,1]

    t_end = cfg["T"] if args.t_end is None else args.t_end
    #timesteps = torch.arange(args.t_start, t_end, args.t_stride, device=device).long()
    timesteps = torch.tensor(
    np.unique(
        np.logspace(np.log10(args.t_start), np.log10(t_end - 1), 8).astype(int)
    ),
    device=device,
    dtype=torch.long
)

    # Validate weights if provided
    if args.weights is not None and len(args.weights) != len(args.mus):
        raise ValueError(f"--weights length {len(args.weights)} must match --mus length {len(args.mus)}")

    # Collect curves
    gt_curves = []
    learned_curves = []

    for t0 in timesteps.tolist():
        t_batch = torch.full((X.shape[0],), int(t0), device=device, dtype=torch.long)  # [N]

        # Ground-truth marginal MoG score at x_t = X
        s_true = mog_score_closed_form(
            X, diffusion, t0, mus=tuple(args.mus), sigma=args.sigma, weights=args.weights
        )  # [N,1]

        # Learned score from epsilon prediction: s_hat(x,t) = -eps_theta(x,t) / sqrt(1 - alpha_bar_t)
        eps_pred = model(X, t_batch)  # [N,1]
        denom = diffusion._extract(diffusion.sqrt_one_minus_alphas_bar, t_batch, X.shape)  # [N,1]
        s_hat = -eps_pred / (denom + 1e-30)  # [N,1]

        gt_curves.append(s_true.squeeze(1))
        learned_curves.append(s_hat.squeeze(1))

    gt_curves = torch.stack(gt_curves, dim=0)          # [nT, N]
    learned_curves = torch.stack(learned_curves, dim=0) # [nT, N]

    # ---- Plot: two subplots with legends ----
    x_np = X[:, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax0, ax1 = axes

    ax0.set_title("Ground Truth Score (closed form)")
    ax1.set_title("Learned Score Function (from DDPM ε-prediction)")

    ax0.set_xlabel("x")
    ax1.set_xlabel("x")
    ax0.set_ylabel("score")
    ax1.set_ylabel("score")

    for i, t0 in enumerate(timesteps.tolist()):
        y0 = gt_curves[i].detach().cpu().numpy()
        y1 = learned_curves[i].detach().cpu().numpy()
        ax0.plot(x_np, y0, label=f"t={t0}")
        ax1.plot(x_np, y1, label=f"t={t0}")

    ax0.legend(fontsize=8)
    ax1.legend(fontsize=8)

    fig.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()