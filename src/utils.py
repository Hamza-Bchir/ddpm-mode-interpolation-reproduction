import torch
import random
import numpy as np
from torch.utils.data import Dataset
import itertools


def set_seed(seed: int = 0) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import math
import torch
import torch.nn.functional as F


def sinusoidal_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10_000
) -> torch.Tensor:
    """
    Compute sinusoidal timestep embeddings as commonly used in diffusion models.
    """
    if t.dim() != 1:
        raise ValueError(f"`t` must be 1D of shape (B,), got {tuple(t.shape)}")

    if dim < 2:
        raise ValueError(f"`dim` must be >= 2, got {dim}")

    device = t.device
    B = t.shape[0]

    half_dim = dim // 2
    if half_dim < 1:
        raise ValueError(f"`dim` must allow at least one sine/cosine pair, got {dim}")

    # Frequencies: exp(-log(max_period) * i / (half_dim - 1))
    # Handles half_dim == 1 safely
    if half_dim == 1:
        freqs = torch.ones(1, device=device, dtype=torch.float32)
    else:
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half_dim)

    # If dim is odd, pad one dimension
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb



class GaussianDataset(Dataset):
    """
    Dataset representing a Gaussian distribution or a mixture of Gaussians.

    This dataset generates synthetic samples drawn from either:
        - A single Gaussian distribution
        - A mixture of multiple Gaussian components with uniform weights

    It supports:
        - Arbitrary number of mixture components
        - Configurable means and standard deviations
        - Shared standard deviation across components
        - Preset configurations for common 1D and 2D cases

    Preset Behavior (if preset=True):
        - dim=1:
            * 3 components
            * Means: [[1.0], [2.0], [3.0]]
            * Shared std: 0.05

        - dim=2:
            * 25 components arranged on a 5×5 equally spaced grid
            * Grid bounds controlled by grid_min_2d and grid_max_2d
            * Shared std controlled by std_2d

    Each sample consists of:
        - x: data point drawn from selected Gaussian component
        - y: integer component index

    Attributes:
        n (int):
            Total number of samples.

        dim (int):
            Dimensionality of each data point.

        n_components (int):
            Number of Gaussian mixture components.

        means (torch.Tensor):
            Tensor of shape (n_components, dim) containing component means.

        stds (torch.Tensor):
            Tensor of shape (n_components, dim) containing component standard deviations.
    """

    def __init__(
        self,
        n: int,
        dim: int,
        n_components: int | None = None,
        means=None,
        stds=None,
        shared_std: bool = True,
        preset: bool = True,
        grid_size_2d: int = 5,
        grid_min_2d: float = -2.0,
        grid_max_2d: float = 2.0,
        std_2d: float = 0.05,
    ) -> None:
        """
        Initialize the GaussianDataset.

        Args:
            n (int):
                Total number of samples to generate.

            dim (int):
                Dimensionality of each sample.

            n_components (int | None, optional):
                Number of Gaussian mixture components.
                Required if preset=False and means are not provided.

            means (array-like or torch.Tensor, optional):
                Mean vectors for each component.
                Expected shape: (n_components, dim).

            stds (float, int, array-like, or torch.Tensor, optional):
                Standard deviations for each component.
                If shared_std=True, must be a scalar.
                Otherwise, must match shape (n_components, dim).

            shared_std (bool, optional):
                If True, a single scalar std is shared across all components.
                If False, each component may have its own std.
                Default is True.

            preset (bool, optional):
                If True, automatically constructs predefined mixtures for dim=1 or dim=2.
                Default is True.

            grid_size_2d (int, optional):
                Number of grid points per axis for 2D preset.
                Default is 5.

            grid_min_2d (float, optional):
                Minimum coordinate value for 2D grid preset.
                Default is -2.0.

            grid_max_2d (float, optional):
                Maximum coordinate value for 2D grid preset.
                Default is 2.0.

            std_2d (float, optional):
                Shared standard deviation used in 2D preset.
                Default is 0.2.
        """
        super().__init__()
        self.n = int(n)
        self.dim = int(dim)

        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")

        # ---- presets
        if preset:
            if self.dim == 1:
                n_components = 3
                means = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
                stds = 0.05
                shared_std = True
            elif self.dim == 2:
                coords = torch.linspace(grid_min_2d, grid_max_2d, grid_size_2d)
                grid = list(itertools.product(coords.tolist(), coords.tolist()))
                means = torch.tensor(grid, dtype=torch.float32)  # (grid_size^2, 2)
                n_components = int(means.size(0))
                stds = float(std_2d)
                shared_std = True
            else:
                raise ValueError("preset=True is only implemented for dim=1 or dim=2")

        if means is None:
            raise ValueError("means must be provided when preset=False")

        means = torch.as_tensor(means, dtype=torch.float32)
        if means.dim() == 1:
            means = means.unsqueeze(1)  # (K,) -> (K,1)

        if means.dim() != 2:
            raise ValueError(f"means must be 2D after conversion, got shape {tuple(means.shape)}")

        if means.shape[1] != self.dim:
            raise ValueError(
                f"Means dimension mismatch: expected dim={self.dim}, got {means.shape[1]}"
            )

        if n_components is not None and int(n_components) != means.shape[0]:
            raise ValueError(
                f"n_components={n_components} but means has {means.shape[0]} components"
            )

        self.n_components = int(means.shape[0])
        self.means = means

        # ---- stds handling
        if shared_std:
            if not isinstance(stds, (float, int)):
                raise ValueError("When shared_std=True, stds must be a scalar float/int")
            stds_t = torch.full(
                (self.n_components, self.dim),
                float(stds),
                dtype=torch.float32,
            )
        else:
            if stds is None:
                raise ValueError("stds must be provided when shared_std=False")
            stds_t = torch.as_tensor(stds, dtype=torch.float32)
            if stds_t.dim() == 1:
                # (K,) or (K,1) style -> expand to (K, dim)
                stds_t = stds_t.unsqueeze(1).repeat(1, self.dim)
            if stds_t.shape != self.means.shape:
                raise ValueError(
                    f"stds must match means shape {tuple(self.means.shape)} when shared_std=False, "
                    f"got {tuple(stds_t.shape)}"
                )

        self.stds = stds_t

        # ---- sample mixture with uniform weights
        comp_ids = torch.randint(0, self.n_components, (self.n,), dtype=torch.long)
        noise = torch.randn(self.n, self.dim, dtype=torch.float32)
        self.x = self.means[comp_ids] + self.stds[comp_ids] * noise
        self.y = comp_ids  # component id

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.n

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            (x, y) where:
                x: Tensor of shape (dim,)
                y: Long tensor scalar (component index)
        """
        return self.x[idx], self.y[idx]