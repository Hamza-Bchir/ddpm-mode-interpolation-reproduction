import math
import torch


def beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """
    Generate a beta schedule for a diffusion process.

    The beta schedule defines the variance of the forward diffusion
    process at each timestep. Different schedules influence training
    stability and sample quality.

    Args:
        T (int):
            Total number of diffusion timesteps. Must be strictly positive.

        beta_start (float, optional):
            Initial beta value (used in schedules that require a range).
            Default is 1e-4.

        beta_end (float, optional):
            Final beta value (used in schedules that require a range).
            Default is 2e-2.

        schedule_type (str, optional):
            Type of schedule to generate. Supported values may include:

            - "linear": Linearly spaced betas from beta_start to beta_end.
            - "cosine": Cosine schedule based on a cumulative alpha_bar
              formulation (e.g., Nichol & Dhariwal style).

            Default is "linear".

    Returns:
        torch.Tensor:
            A 1D tensor of shape (T,) containing the beta values
            for each diffusion timestep. The tensor is typically
            of dtype torch.float32.
    """
    if T <= 0:
        raise ValueError("T must be positive")

    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    elif schedule_type == "cosine":
        # Cosine schedule (Nichol & Dhariwal style) implemented via alpha_bar
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(1e-8, 0.999).float()

    else:
        raise NotImplementedError(f"Unknown schedule_type='{schedule_type}'")

    return betas


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) with epsilon-prediction.

    Implements:
        - Forward diffusion q(x_t | x_0)
        - Reverse sampling p_theta(x_{t-1} | x_t)
        - Full sampling loop

    Uses the standard noise-prediction parameterization.
    """

    def __init__(
        self,
        betas: torch.Tensor | None = None,
        T: int = 1000,
        device: str | torch.device = "cuda",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear",
    ) -> None:
        """
        Initialize diffusion process.

        Args:
            betas: Optional precomputed beta schedule of shape (T,).
            T: Number of diffusion timesteps.
            device: Device for tensors.
            beta_start: Starting beta value (if schedule generated internally).
            beta_end: Final beta value (if schedule generated internally).
            schedule_type: Type of beta schedule ("linear", "cosine", ...).
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.T = int(T)

        if betas is None:
            betas = beta_schedule(
                self.T,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule_type=schedule_type,
            )
        else:
            betas = torch.as_tensor(betas, dtype=torch.float32)
            if betas.numel() != self.T:
                raise ValueError(f"betas must have shape (T,), got {tuple(betas.shape)} vs T={self.T}")

        self.betas = betas.to(self.device)                      # (T,)
        self.alphas = (1.0 - self.betas).to(self.device)        # (T,)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)     # (T,)

        # Precompute useful terms
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)                      # (T,)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)      # (T,)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)                  # (T,)

        # Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alphas_bar_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_bar[:-1]], dim=0
        )  # (T,)
        self.posterior_variance = self.betas * (1.0 - alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Extract coefficients indexed by batch timesteps and reshape
        for broadcasting over input tensor.

        Args:
            a: Tensor of shape (T,).
            t: Tensor of shape (B,) containing timestep indices.
            x_shape: Shape of target tensor.

        Returns:
            Tensor reshaped to broadcast over x.
        """
        if t.dim() != 1:
            raise ValueError(f"t must be (B,), got {tuple(t.shape)}")

        out = a.gather(0, t)  # (B,)
        return out.view(t.size(0), *([1] * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step.

        Args:
            x0: Clean input sample (B, ...).
            t: Timestep indices (B,).
            noise: Optional noise tensor (B, ...).

        Returns:
            x_t: Noised sample.
            noise: Noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        One reverse diffusion step.

        Args:
            model: Noise prediction model.
            x_t: Current noisy sample (B, ...).
            t: Current timestep indices (B,).

        Returns:
            x_{t-1}: Denoised sample.
            \hat{eps}: Predicted noise (useful for figure 6). 
        """
        eps_pred = model(x_t, t)  # predicts noise

        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_1mab_t = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)

        # mean (epsilon-parameterization)
        mu = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_1mab_t) * eps_pred)

        # variance (posterior)
        var = self._extract(self.posterior_variance, t, x_t.shape)

        # no noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.size(0), *([1] * (x_t.dim() - 1)))
        x_prev = mu + nonzero_mask * torch.sqrt(var) * noise
        return x_prev, eps_pred

    @torch.no_grad()
    def sample(
        self,
        model,
        n: int,
        shape: int | tuple,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """
        Full reverse sampling loop.

        Args:
            model: Noise prediction model.
            n: Number of samples.
            shape: Sample shape excluding batch.
            device: Optional override device.

        Returns:
            Generated samples.
        """
        if device is None:
            device = self.device
        device = torch.device(device) if isinstance(device, str) else device

        if isinstance(shape, int):
            sample_shape = (n, shape)
        else:
            sample_shape = (n, *shape)

        x = torch.randn(sample_shape, device=device)

        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            x, _ = self.p_sample(model, x, t)

        return x